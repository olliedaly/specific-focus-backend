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
from google.cloud import firestore
import asyncio
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor

# --- Basic Configuration ---
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Focus Monitor Pro Backend", version="0.2.0")

# --- Firestore Configuration ---
try:
    db = firestore.Client()
    logger.info("Firestore client initialized successfully.")
    # You might want to specify project and database_id if not relying on environment defaults
    # db = firestore.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"), database=os.getenv("FIRESTORE_DATABASE_ID"))
except DefaultCredentialsError:
    logger.error("CRITICAL: Could not find default Google Cloud credentials for Firestore. Ensure Application Default Credentials are set.")
    db = None # Set db to None so app can still start (though it won't work correctly)
except Exception as e:
    logger.error(f"CRITICAL: Error initializing Firestore client: {e}")
    db = None

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

TOKEN_CACHE = {}  # Simple in-memory cache for tokens
TOKEN_CACHE_TTL = 300  # 5 minutes
USER_STATUS_CACHE = {}  # Cache user status
USER_STATUS_CACHE_TTL = 60  # 1 minute

# Thread pool for background tasks
background_executor = ThreadPoolExecutor(max_workers=2)

@lru_cache(maxsize=128)
def get_cached_model():
    """Cache the Gemini model instance to avoid recreation overhead"""
    return genai.GenerativeModel(
        MODEL_NAME,
        generation_config=generation_config
        # No safety settings - just binary focus classification
    )

def validate_chrome_extension_token(access_token):
    if not access_token:
        return None, "No access token provided", 401 
    
    token_info_url = f"https://www.googleapis.com/oauth2/v3/tokeninfo?access_token={access_token}"
    try:
        response = requests.get(token_info_url)
        
        if response.status_code != 200:
             error_detail = "Unknown token validation error"
             try:
                 error_json = response.json()
                 error_detail = error_json.get('error_description', error_json.get('error', response.text))
             except ValueError:
                 error_detail = response.text
             logger.warning(f"Token validation failed with status {response.status_code}: {error_detail}")
             return None, f"Token validation failed: {error_detail}", response.status_code

        token_info = response.json()

        if 'error' in token_info: 
            error_msg = f"Token validation error: {token_info.get('error_description', token_info['error'])}"
            logger.warning(error_msg)
            return None, error_msg, 401

        if token_info.get('aud') != CHROME_EXTENSION_OAUTH_CLIENT_ID:
            logger.warning(f"Token audience mismatch. Expected: {CHROME_EXTENSION_OAUTH_CLIENT_ID}, Got: {token_info.get('aud')}")
            return None, "Token audience mismatch. Token was not intended for this application.", 403

        logger.info(f"Token validated successfully for email: {token_info.get('email')}, user_id: {token_info.get('sub')}")
        return token_info.get('sub'), None, 200
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during token validation: {e}")
        return None, f"Network error during token validation: {e}", 503
    except ValueError as e:
        logger.error(f"JSON decoding error during token validation: {e}. Response text: {response.text if 'response' in locals() else 'N/A'}")
        return None, "Error decoding token validation response.", 500
    except Exception as e:
        logger.error(f"An unexpected error occurred during token validation: {e}", exc_info=True)
        return None, f"An unexpected error occurred during token validation: {e}", 500

def validate_chrome_extension_token_cached(access_token):
    """Cached version of token validation"""
    if not access_token:
        return None, "No access token provided", 401
    
    # Check cache first
    cache_key = access_token[:20] + "..." + access_token[-10:]  # Partial token for security
    current_time = time.time()
    
    if cache_key in TOKEN_CACHE:
        cached_data, timestamp = TOKEN_CACHE[cache_key]
        if current_time - timestamp < TOKEN_CACHE_TTL:
            logger.info("Token validation served from cache")
            return cached_data
    
    # Fall back to original validation
    result = validate_chrome_extension_token(access_token)
    
    # Cache successful validations only
    if result[1] is None:  # No error
        TOKEN_CACHE[cache_key] = (result, current_time)
        # Clean old entries periodically
        if len(TOKEN_CACHE) > 200:
            old_keys = [k for k, (_, ts) in TOKEN_CACHE.items() if current_time - ts > TOKEN_CACHE_TTL]
            for k in old_keys[:50]:  # Remove oldest 50
                del TOKEN_CACHE[k]
    
    return result

def get_user_status_cached(user_id: str):
    """Cached version of user status lookup"""
    current_time = time.time()
    
    if user_id in USER_STATUS_CACHE:
        cached_status, timestamp = USER_STATUS_CACHE[user_id]
        if current_time - timestamp < USER_STATUS_CACHE_TTL:
            logger.info(f"User status served from cache for user {user_id}")
            return cached_status
    
    # Fetch fresh data
    status = get_or_create_user_status(user_id)
    
    # Cache it
    USER_STATUS_CACHE[user_id] = (status, current_time)
    
    return status

def background_increment_api_count(user_status: dict):
    """Run API count increment in background"""
    def _increment():
        try:
            increment_api_call_count(user_status)
        except Exception as e:
            logger.error(f"Background API count increment failed: {e}")
    
    # Run in thread pool since Firestore operations are blocking
    import threading
    thread = threading.Thread(target=_increment)
    thread.daemon = True
    thread.start()

MODEL_NAME = "gemini-2.0-flash-lite"

generation_config = genai.types.GenerationConfig(
    candidate_count=1,
    temperature=0.1, # Low temperature for deterministic classification
    max_output_tokens=5, # OPTIMIZED: Limit to just "Relevant" or "Irrelevant"
    top_p=0.1,  # Very focused responses
    top_k=2     # Only consider top 2 tokens
)

# No safety settings needed for simple focus classification

FREE_API_CALL_LIMIT = 50
USERS_COLLECTION = "users" # Firestore collection name

def get_or_create_user_status(user_id: str):
    if not db:
        logger.error(f"Firestore client not available. Cannot process user {user_id}.")
        # Depending on policy, could raise an exception or return a default "deny"
        raise HTTPException(status_code=503, detail="Backend database service is unavailable.")

    user_ref = db.collection(USERS_COLLECTION).document(user_id)
    user_doc = user_ref.get() # Standard Firestore client is synchronous

    if user_doc.exists:
        user_data = user_doc.to_dict()
        return {
            "user_id": user_id,
            "is_premium": user_data.get("is_premium", False),
            "api_request_count": user_data.get("api_request_count", 0),
            "doc_ref": user_ref # Return reference for easy updates
        }
    else:
        logger.info(f"New user detected: {user_id}. Creating Firestore document.")
        new_user_data = {
            "user_id": user_id,
            "is_premium": False,
            "api_request_count": 0,
            "created_at": firestore.SERVER_TIMESTAMP # Good to have a creation timestamp
        }
        user_ref.set(new_user_data) # Standard Firestore client is synchronous
        return {
            "user_id": user_id,
            "is_premium": False,
            "api_request_count": 0,
            "doc_ref": user_ref
        }

def increment_api_call_count(user_status: dict):
    if not db:
        logger.error(f"Firestore client not available. Cannot increment count for user {user_status['user_id']}.")
        return # Or raise

    user_ref = user_status["doc_ref"]
    try:
        # Atomically increment the count
        user_ref.update({"api_request_count": firestore.Increment(1)})
        logger.info(f"Incremented API call count for user {user_status['user_id']}.")
    except Exception as e:
        logger.error(f"Error incrementing API call count for user {user_status['user_id']}: {e}")
        # Decide on error handling: retry, log, etc.

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
    if not db:
        logger.error("Firestore client not initialized. /classify endpoint cannot function.")
        raise HTTPException(status_code=503, detail="Backend database service is unavailable for /classify.")

    # Extract token for parallel processing
    auth_header = fastapi_request.headers.get('Authorization')
    access_token = None
    if auth_header and auth_header.startswith('Bearer '):
        access_token = auth_header.split('Bearer ')[1]
    
    # Validate token (cached)
    user_id, error_message, status_code = validate_chrome_extension_token_cached(access_token)
    if error_message:
        raise HTTPException(status_code=status_code, detail=error_message)
    
    logger.info(f"Request to /classify authenticated for user_id: {user_id}. URL: {data.url}, Focus: {data.session_focus}")
    
    # Get user status (cached)
    user_status = get_user_status_cached(user_id)
    
    # Check limits
    if not user_status["is_premium"]:
        if user_status["api_request_count"] >= FREE_API_CALL_LIMIT:
            logger.warning(f"User {user_id} has reached free API call limit.")
            raise HTTPException(
                status_code=402,
                detail={
                    "message": "You have exceeded your free API request limit. Please upgrade to continue.",
                    "limit_reached": True,
                    "current_count": user_status["api_request_count"],
                    "limit": FREE_API_CALL_LIMIT
                }
            )
    
    # Build prompt
    prompt = build_gemini_prompt(data)
    
    try:
        # Use cached model instance
        model = get_cached_model()
        response = await model.generate_content_async(prompt)

        raw_text_response = response.text.strip().lower()
        logger.info(f"Gemini raw response for {data.url}: '{raw_text_response}'")

        if "relevant" in raw_text_response and "irrelevant" not in raw_text_response:
            assessment = "Relevant"
        elif "irrelevant" in raw_text_response:
            assessment = "Irrelevant"
        else:
            logger.warning(f"Gemini unclear response for {data.url}. Got: '{response.text.strip()}'. Defaulting to Irrelevant.")
            assessment = "Irrelevant"
        
        # BACKGROUND TASK: Increment API call count asynchronously (don't wait)
        if not user_status["is_premium"]:
            background_increment_api_count(user_status)
            # Invalidate user cache to ensure next request gets fresh count
            if user_id in USER_STATUS_CACHE:
                del USER_STATUS_CACHE[user_id]

        return ClassificationResponse(assessment=assessment, original_url=data.url)

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error during Gemini API call or processing for {data.url}: {e}", exc_info=True)
        if hasattr(e, 'message') and "API key not valid" in e.message:
             raise HTTPException(status_code=401, detail=f"Gemini API Key Error: {e.message}")
        if 'response' in locals() and response and response.prompt_feedback and response.prompt_feedback.block_reason:
            logger.warning(f"Prompt for {data.url} was blocked. Reason: {response.prompt_feedback.block_reason}")
            return ClassificationResponse(assessment="Error", original_url=data.url)

        raise HTTPException(status_code=503, detail=f"Error communicating with Gemini API: {str(e)}")

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