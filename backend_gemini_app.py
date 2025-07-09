from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
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
import stripe

# --- Basic Configuration ---
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Focus Monitor Pro Backend", version="0.2.0")

# Templates configuration
templates = Jinja2Templates(directory="templates")

# Stripe configuration
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
    logger.info("Stripe API configured successfully.")
else:
    logger.warning("STRIPE_SECRET_KEY not found. Payment features will not work.")

if not STRIPE_PRICE_ID:
    logger.warning("STRIPE_PRICE_ID not found in environment. Payment checkout will fail.")

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

origins = [
    "https://olliedaly.github.io" # Allow the upgrade page to make requests
]
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
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
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

@app.get("/user-status")
async def get_user_status(fastapi_request: Request):
    """Get user's current status including API call count and premium status"""
    auth_header = fastapi_request.headers.get('Authorization')
    access_token = None
    if auth_header and auth_header.startswith('Bearer '):
        access_token = auth_header.split('Bearer ')[1]
    
    user_id, error_message, status_code = validate_chrome_extension_token_cached(access_token)
    if error_message:
        raise HTTPException(status_code=status_code, detail=error_message)
    
    user_status = get_user_status_cached(user_id)
    
    return {
        "user_id": user_id,
        "is_premium": user_status["is_premium"],
        "api_request_count": user_status["api_request_count"],
        "limit": FREE_API_CALL_LIMIT
    }

class SubscriptionUpdate(BaseModel):
    purchase_token: str
    sku: str
    user_email: str

@app.post("/update-subscription")
async def update_subscription(data: SubscriptionUpdate, fastapi_request: Request):
    """Update user's premium status based on Chrome Web Store purchase"""
    auth_header = fastapi_request.headers.get('Authorization')
    access_token = None
    if auth_header and auth_header.startswith('Bearer '):
        access_token = auth_header.split('Bearer ')[1]
    
    user_id, error_message, status_code = validate_chrome_extension_token_cached(access_token)
    if error_message:
        raise HTTPException(status_code=status_code, detail=error_message)
    
    try:
        # In a production system, you would verify the purchase_token with Google's API
        # For now, we'll trust the client (this should be improved for production)
        logger.info(f"Updating subscription for user {user_id} with purchase token: {data.purchase_token[:20]}...")
        
        # Update user to premium status
        user_ref = db.collection(USERS_COLLECTION).document(user_id)
        user_ref.update({
            "is_premium": True,
            "subscription_sku": data.sku,
            "purchase_token": data.purchase_token,
            "subscription_updated_at": firestore.SERVER_TIMESTAMP
        })
        
        # Invalidate cache
        if user_id in USER_STATUS_CACHE:
            del USER_STATUS_CACHE[user_id]
        
        logger.info(f"Successfully updated user {user_id} to premium status")
        
        return {
            "status": "success",
            "message": "Subscription updated successfully",
            "is_premium": True
        }
        
    except Exception as e:
        logger.error(f"Error updating subscription for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update subscription")

@app.get("/upgrade", response_class=HTMLResponse)
async def upgrade_page(request: Request, token: str = None, user_id: str = None):
    """Serve the upgrade page with user token"""
    try:
        return templates.TemplateResponse("upgrade.html", {
            "request": request,
            "stripe_publishable_key": STRIPE_PUBLISHABLE_KEY,
            "user_token": token,
            "user_id": user_id
        })
    except Exception as e:
        logger.error(f"Error rendering upgrade template: {e}")
        # Fallback to inline HTML if template fails
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Upgrade to Specific Focus Premium</title>
            <script src="https://js.stripe.com/v3/"></script>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 40px 20px;
                    background: #f8f9fa;
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 12px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                h1 {{
                    color: #333;
                    margin-bottom: 10px;
                }}
                .subtitle {{
                    color: #666;
                    font-size: 18px;
                    margin-bottom: 30px;
                }}
                .price {{
                    font-size: 48px;
                    color: #6A8A7B;
                    font-weight: bold;
                    margin: 20px 0;
                }}
                .price-detail {{
                    color: #666;
                    margin-bottom: 30px;
                }}
                .features {{
                    text-align: left;
                    margin: 30px 0;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 8px;
                }}
                .feature {{
                    margin: 10px 0;
                    display: flex;
                    align-items: center;
                }}
                .feature::before {{
                    content: "✓";
                    color: #6A8A7B;
                    font-weight: bold;
                    margin-right: 10px;
                }}
                #checkout-button {{
                    background: #6A8A7B;
                    color: white;
                    border: none;
                    padding: 15px 40px;
                    font-size: 18px;
                    border-radius: 8px;
                    cursor: pointer;
                    margin-top: 20px;
                    transition: background 0.2s;
                }}
                #checkout-button:hover {{
                    background: #597567;
                }}
                #checkout-button:disabled {{
                    background: #ccc;
                    cursor: not-allowed;
                }}
                .error {{
                    color: #d32f2f;
                    margin-top: 20px;
                }}
                .loading {{
                    display: none;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Upgrade to Premium</h1>
                <p class="subtitle">Unlock unlimited AI-powered focus monitoring</p>
                
                <div class="price">$4.99</div>
                <div class="price-detail">per month</div>
                
                <div class="features">
                    <div class="feature">Unlimited AI classifications per month</div>
                    <div class="feature">Priority support</div>
                    <div class="feature">Advanced analytics (coming soon)</div>
                    <div class="feature">No daily usage limits</div>
                </div>
                
                <button id="checkout-button">Start Premium Subscription</button>
                
                <div id="loading" class="loading">
                    Processing your payment...
                </div>
                
                <div id="error-message" class="error"></div>
                
                <p style="font-size: 12px; color: #999; margin-top: 30px;">
                    Cancel anytime from your extension settings. Secure payments powered by Stripe.
                </p>
            </div>

            <script>
                const userToken = '{token or ""}';
                const userId = '{user_id or ""}';
                
                if (!userToken || !userId || userToken === 'None' || userId === 'None') {{
                    document.getElementById('error-message').textContent = 'Invalid access. Please try again from your extension.';
                    document.getElementById('checkout-button').disabled = true;
                }}

                const stripe = Stripe('{STRIPE_PUBLISHABLE_KEY or ""}');
                
                document.getElementById('checkout-button').addEventListener('click', async () => {{
                    const button = document.getElementById('checkout-button');
                    const loading = document.getElementById('loading');
                    const errorDiv = document.getElementById('error-message');
                    
                    button.disabled = true;
                    loading.style.display = 'block';
                    errorDiv.textContent = '';
                    
                    try {{
                        const response = await fetch('/create-checkout-session', {{
                            method: 'POST',
                            headers: {{
                                'Content-Type': 'application/json',
                                'Authorization': `Bearer ${{userToken}}`
                            }},
                            body: JSON.stringify({{
                                user_id: userId
                            }})
                        }});
                        
                        if (!response.ok) {{
                            throw new Error('Failed to create checkout session');
                        }}
                        
                        const session = await response.json();
                        
                        const result = await stripe.redirectToCheckout({{
                            sessionId: session.id
                        }});
                        
                        if (result.error) {{
                            throw new Error(result.error.message);
                        }}
                        
                    }} catch (error) {{
                        console.error('Payment error:', error);
                        errorDiv.textContent = error.message || 'Payment failed. Please try again.';
                        button.disabled = false;
                        loading.style.display = 'none';
                    }}
                }});
            </script>
        </body>
        </html>
        """)

class CheckoutSession(BaseModel):
    user_id: str
    success_url: str | None = None
    cancel_url: str | None = None

@app.post("/create-checkout-session")
async def create_checkout_session(data: CheckoutSession, fastapi_request: Request):
    """Create a Stripe checkout session"""
    if not STRIPE_SECRET_KEY or not STRIPE_PRICE_ID:
        raise HTTPException(status_code=503, detail="Payment system not configured on the server")
    
    auth_header = fastapi_request.headers.get('Authorization')
    access_token = None
    if auth_header and auth_header.startswith('Bearer '):
        access_token = auth_header.split('Bearer ')[1]
    
    user_id, error_message, status_code = validate_chrome_extension_token_cached(access_token)
    if error_message:
        raise HTTPException(status_code=status_code, detail=error_message)
    
    try:
        # Create customer or retrieve existing
        user_status = get_user_status_cached(user_id)
        
        # Get user email from token
        token_info_url = f"https://www.googleapis.com/oauth2/v3/tokeninfo?access_token={access_token}"
        response = requests.get(token_info_url)
        token_info = response.json()
        user_email = token_info.get('email')
        
        checkout_session = stripe.checkout.Session.create(
            customer_email=user_email,
            payment_method_types=['card'],
            line_items=[{
                'price': STRIPE_PRICE_ID,
                'quantity': 1,
            }],
            mode='subscription',
            success_url=data.success_url or f'https://specific-focus-backend-1056415616503.europe-west1.run.app/payment-success?session_id={{CHECKOUT_SESSION_ID}}',
            cancel_url=data.cancel_url or f'https://olliedaly.github.io/specific-focus-extension/upgrade.html?token={access_token}&user_id={user_id}',
            metadata={
                'user_id': user_id,
                'user_email': user_email
            }
        )
        
        return {"id": checkout_session.id}
        
    except Exception as e:
        logger.error(f"Error creating checkout session for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create checkout session")

@app.get("/payment-success", response_class=HTMLResponse)
async def payment_success(request: Request, session_id: str = None):
    """Handle successful payment"""
    return """
    <html>
    <head><title>Payment Successful</title></head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; text-align: center; padding: 40px;">
        <h1 style="color: #6A8A7B;">Payment Successful!</h1>
        <p>Your Specific Focus Premium subscription is now active.</p>
        <p>You can close this tab and return to your extension.</p>
        <script>
            // Auto-close after 3 seconds
            setTimeout(() => window.close(), 3000);
        </script>
    </body>
    </html>
    """

@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhooks for subscription events"""
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Webhook not configured")
    
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        logger.error(f"Invalid payload in webhook: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature in webhook: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        user_id = session['metadata']['user_id']
        
        # Update user to premium
        try:
            user_ref = db.collection(USERS_COLLECTION).document(user_id)
            user_ref.update({
                "is_premium": True,
                "stripe_customer_id": session.get('customer'),
                "stripe_subscription_id": session.get('subscription'),
                "subscription_activated_at": firestore.SERVER_TIMESTAMP
            })
            
            # Invalidate cache
            if user_id in USER_STATUS_CACHE:
                del USER_STATUS_CACHE[user_id]
            
            logger.info(f"Successfully activated premium for user {user_id}")
        except Exception as e:
            logger.error(f"Error activating premium for user {user_id}: {e}")
    
    elif event['type'] == 'customer.subscription.deleted':
        # Handle subscription cancellation
        subscription = event['data']['object']
        customer_id = subscription['customer']
        
        # Find user by customer ID and deactivate premium
        try:
            users_query = db.collection(USERS_COLLECTION).where("stripe_customer_id", "==", customer_id).limit(1)
            docs = users_query.stream()
            
            for doc in docs:
                doc.reference.update({
                    "is_premium": False,
                    "subscription_cancelled_at": firestore.SERVER_TIMESTAMP
                })
                
                # Invalidate cache
                if doc.id in USER_STATUS_CACHE:
                    del USER_STATUS_CACHE[doc.id]
                
                logger.info(f"Successfully deactivated premium for user {doc.id}")
                break
        except Exception as e:
            logger.error(f"Error deactivating premium for customer {customer_id}: {e}")
    
    return {"status": "success"}

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