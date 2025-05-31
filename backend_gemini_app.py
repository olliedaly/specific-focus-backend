from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import uvicorn
import os
from dotenv import load_dotenv
import logging
import requests
from google.auth.exceptions import DefaultCredentialsError

# --- Basic Configuration ---
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Focus Monitor Pro Backend", version="0.2.0")

# --- CORS Configuration ---
ALLOWED_ORIGIN_EXTENSION_ID = os.getenv("ALLOWED_ORIGIN_EXTENSION_ID")

origins = []
# Development mode: Allow all if ID is not set or explicitly "*"
# In a real production environment, you'd remove this permissive block.
DEVELOPMENT_MODE_ALLOW_ALL_ORIGINS = os.getenv("DEVELOPMENT_MODE_ALLOW_ALL_ORIGINS", "false").lower() == "true"

if ALLOWED_ORIGIN_EXTENSION_ID and ALLOWED_ORIGIN_EXTENSION_ID != "*":
    if not ALLOWED_ORIGIN_EXTENSION_ID.startswith("chrome-extension://"):
        logger.warning(f"ALLOWED_ORIGIN_EXTENSION_ID '{ALLOWED_ORIGIN_EXTENSION_ID}' does not start with 'chrome-extension://'. This might cause CORS issues if it's not a wildcard.")
    origins.append(ALLOWED_ORIGIN_EXTENSION_ID)
    logger.info(f"CORS configured for specific origin: {ALLOWED_ORIGIN_EXTENSION_ID}")
elif DEVELOPMENT_MODE_ALLOW_ALL_ORIGINS:
    origins = ["*"]
    logger.warning("DEVELOPMENT_MODE_ALLOW_ALL_ORIGINS is true. Allowing all origins for CORS. DO NOT USE IN PRODUCTION.")
else:
    # Default behavior if no extension ID and dev mode not explicitly enabled: still restrictive
    logger.error("CRITICAL: No specific CORS origin configured (ALLOWED_ORIGIN_EXTENSION_ID is not set or is '*') AND DEVELOPMENT_MODE_ALLOW_ALL_ORIGINS is not 'true'. Extension calls will likely be blocked.")
    # origins will remain empty, effectively blocking. Or you could choose a default restrictive behavior.
    # For now, let's keep it empty to highlight the issue.

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else [], # Pass the list of origins; empty list blocks all.
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# --- Google Gemini Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("CRITICAL: GEMINI_API_KEY not found in environment variables. The API will not work.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Google Gemini API configured successfully.")
    except Exception as e:
        logger.error(f"Error configuring Google Gemini API: {e}")

# --- OAuth Token Validation ---
CHROME_EXTENSION_OAUTH_CLIENT_ID = "1056415616503-jagk9ejnakcq643so5dsfvsoagjppfac.apps.googleusercontent.com"

def validate_chrome_extension_token(access_token):
    if not access_token:
        # Return None for user_id, error message, and status code
        return None, "No access token provided", 401 
    
    token_info_url = f"https://www.googleapis.com/oauth2/v3/tokeninfo?access_token={access_token}"
    try:
        response = requests.get(token_info_url)
        
        if response.status_code != 200:
             error_detail = "Unknown token validation error"
             try:
                 error_json = response.json()
                 error_detail = error_json.get('error_description', error_json.get('error', response.text))
             except ValueError: # Not a JSON response
                 error_detail = response.text
             logger.warning(f"Token validation failed with status {response.status_code}: {error_detail}")
             return None, f"Token validation failed: {error_detail}", response.status_code

        token_info = response.json() # Should be safe now after status_code check

        # 'error' field in a 200 OK response is unusual but check just in case
        if 'error' in token_info: 
            error_msg = f"Token validation error: {token_info.get('error_description', token_info['error'])}"
            logger.warning(error_msg)
            return None, error_msg, 401

        if token_info.get('aud') != CHROME_EXTENSION_OAUTH_CLIENT_ID:
            logger.warning(f"Token audience mismatch. Expected: {CHROME_EXTENSION_OAUTH_CLIENT_ID}, Got: {token_info.get('aud')}")
            return None, "Token audience mismatch. Token was not intended for this application.", 403 # Forbidden
        
        # Check for necessary scopes if your backend relies on them
        # required_scopes = {"email", "profile"} # From your manifest
        # token_scopes = set(token_info.get('scope', '').split())
        # if not required_scopes.issubset(token_scopes):
        #     logger.warning(f"Token missing required scopes. Expected: {required_scopes}, Got: {token_scopes}")
        #     return None, "Token does not have the required scopes.", 403

        logger.info(f"Token validated successfully for email: {token_info.get('email')}, user_id: {token_info.get('sub')}")
        return token_info.get('sub'), None, 200 # Return user_id (sub), no error, success status
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during token validation: {e}")
        return None, f"Network error during token validation: {e}", 503 # Service Unavailable
    except ValueError as e: # Handles JSON decoding errors if response is not JSON
        logger.error(f"JSON decoding error during token validation: {e}. Response text: {response.text if 'response' in locals() else 'N/A'}")
        return None, "Error decoding token validation response.", 500
    except Exception as e:
        logger.error(f"An unexpected error occurred during token validation: {e}", exc_info=True)
        return None, f"An unexpected error occurred during token validation: {e}", 500

MODEL_NAME = "gemini-2.0-flash-lite"

generation_config = genai.types.GenerationConfig(
    candidate_count=1,
    temperature=0.1, # Low temperature for deterministic classification
    # max_output_tokens=10 # Maximize efficiency for single word response
)

# Define safety settings to be less restrictive if needed, or adjust as per policy
safety_settings_classification = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]


# --- Pydantic Models ---
class PageClassificationData(BaseModel):
    url: str
    title: str | None = None
    meta_description: str | None = ""
    meta_keywords: str | None = ""
    page_text_snippet: str | None = ""
    session_focus: str

class ClassificationResponse(BaseModel):
    assessment: str # "Relevant" or "Irrelevant" or "Error"
    original_url: str

# --- LLM Prompting ---
def build_gemini_prompt(data: PageClassificationData) -> str:
    content_parts = [
        f"User's current focus: \"{data.session_focus.strip()}\"\n"
    ]
    content_parts.append("Webpage Information:")
    content_parts.append(f"- URL: {data.url.strip()}")
    if data.title and data.title.strip():
        content_parts.append(f"- Title: {data.title.strip()}")
    if data.meta_description and data.meta_description.strip():
        content_parts.append(f"- Meta Description: {data.meta_description.strip()[:500]}") # Truncate meta
    # Meta keywords are often noisy, consider omitting or heavily processing
    # if data.meta_keywords and data.meta_keywords.strip():
    #     content_parts.append(f"- Meta Keywords: {data.meta_keywords.strip()[:200]}")
    if data.page_text_snippet and data.page_text_snippet.strip():
        snippet = data.page_text_snippet.strip()
        content_parts.append(f"- Page Text Snippet (first ~300 chars): {snippet[:300]}...")

    content_parts.append(
        "\nTask: Based ONLY on the information provided above, assess if the webpage is relevant to the user's focus."
    )
    content_parts.append(
        "Respond with a SINGLE WORD: either 'Relevant' or 'Irrelevant'. Do NOT add any other text, explanation, or punctuation."
    )
    return "\n".join(content_parts)

# --- API Endpoint ---
@app.post("/classify", response_model=ClassificationResponse)
async def classify_page_relevance(data: PageClassificationData, fastapi_request: Request):
    if not GEMINI_API_KEY:
        logger.error("Attempted to call /classify endpoint without GEMINI_API_KEY configured.")
        raise HTTPException(status_code=503, detail="Gemini API key not configured on server.")

    # --- Token Validation ---
    auth_header = fastapi_request.headers.get('Authorization')
    access_token = None
    if auth_header and auth_header.startswith('Bearer '):
        access_token = auth_header.split('Bearer ')[1]
    
    user_id, error_message, status_code = validate_chrome_extension_token(access_token)

    if error_message:
        # Use HTTPException for FastAPI error handling
        raise HTTPException(status_code=status_code, detail=error_message) 
    
    # If we reach here, token is valid. user_id contains the Google account's unique 'sub' identifier.
    logger.info(f"Request to /classify authenticated for user_id: {user_id}. URL: {data.url}, Focus: {data.session_focus}")
    # --- End Token Validation ---

    prompt = build_gemini_prompt(data)
    # logger.debug(f"Gemini Prompt:\n{prompt}") # Be cautious logging full prompt if it contains sensitive info

    try:
        model = genai.GenerativeModel(
            MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings_classification
        )
        response = await model.generate_content_async(prompt)

        raw_text_response = response.text.strip().lower()
        logger.info(f"Gemini raw response for {data.url}: '{raw_text_response}'")

        if "relevant" in raw_text_response and "irrelevant" not in raw_text_response: # More robust check
            assessment = "Relevant"
        elif "irrelevant" in raw_text_response:
            assessment = "Irrelevant"
        else:
            logger.warning(f"Gemini did not return a clear 'Relevant' or 'Irrelevant' for {data.url}. Got: '{response.text.strip()}'. Defaulting to Irrelevant.")
            assessment = "Irrelevant" # Default or consider "Error"

        return ClassificationResponse(assessment=assessment, original_url=data.url)

    except Exception as e:
        logger.error(f"Error during Gemini API call or processing for {data.url}: {e}", exc_info=True)
        # Check for specific Gemini API errors if possible, e.g., from genai.types.generation_types.BlockedPromptException
        if hasattr(e, 'message') and "API key not valid" in e.message:
             raise HTTPException(status_code=401, detail=f"Gemini API Key Error: {e.message}")
        # Check for blocked content
        if response and response.prompt_feedback and response.prompt_feedback.block_reason:
            logger.warning(f"Prompt for {data.url} was blocked. Reason: {response.prompt_feedback.block_reason}")
            return ClassificationResponse(assessment="Error", original_url=data.url) # Specific error for blocked

        raise HTTPException(status_code=503, detail=f"Error communicating with Gemini API or processing response: {str(e)}")

@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok", "message": "Backend is running"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    logger.info(f"Starting backend server on host 0.0.0.0 port {port}...")
    if not origins or "*" in origins:
         logger.warning("CORS is configured permissively (either wildcard or no specific origin and dev mode enabled). Ensure this is intended for development only.")
    else:
        logger.info(f"CORS configured for specific origins: {origins}")

    uvicorn.run("backend_gemini_app:app", host="0.0.0.0", port=port, reload=True) # Added reload=True