from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import uvicorn
import os
from dotenv import load_dotenv
import logging

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
async def classify_page_relevance(data: PageClassificationData):
    if not GEMINI_API_KEY:
        logger.error("Attempted to call /classify endpoint without GEMINI_API_KEY configured.")
        raise HTTPException(status_code=503, detail="Gemini API key not configured on server.")

    logger.info(f"Received classification request for URL: {data.url}, Focus: {data.session_focus}")

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