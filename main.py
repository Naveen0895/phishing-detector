

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import joblib
import numpy as np
import re
import urllib.parse
from urllib.parse import urlparse
import logging
from typing import List
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Phishing URL Detector API",
    description="AI-powered API for detecting phishing URLs",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "https://your-frontend-domain.vercel.app",
        # Add your actual frontend domains here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class URLRequest(BaseModel):
    url: str

class PredictionResponse(BaseModel):
    is_phishing: bool
    confidence: float
    explanation: str
    risk_factors: List[str]
    safe_indicators: List[str]

# Feature extraction function
def extract_features(url: str) -> List[float]:
    """Extract features from URL for ML model prediction"""
    try:
        # Parse URL
        parsed = urlparse(url if url.startswith(('http://', 'https://')) else f'https://{url}')
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        query = parsed.query.lower()
        
        features = []
        
        # Length features
        features.append(len(url))  # url_length
        features.append(len(domain))  # domain_length
        features.append(len(path))  # path_length
        features.append(len(query))  # query_length
        
        # Suspicious patterns
        features.append(1 if re.match(r'^\d+\.\d+\.\d+\.\d+', domain) else 0)  # has_ip
        shortening_services = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'short.link']
        features.append(1 if any(service in domain for service in shortening_services) else 0)  # has_shortening
        
        # Suspicious words count
        suspicious_words = ['secure', 'verify', 'suspended', 'update', 'confirm', 'click', 'urgent', 
                          'act-now', 'limited-time', 'expire', 'banking', 'paypal', 'amazon', 'microsoft']
        features.append(sum(1 for word in suspicious_words if word in url.lower()))  # suspicious_words_count
        
        # Domain analysis
        features.append(domain.count('.'))  # subdomain_count
        features.append(domain.count('-'))  # dash_count
        features.append(1 if parsed.scheme == 'https' else 0)  # has_https
        features.append(domain.count('_'))  # underscore_count
        
        # Character analysis
        features.append(sum(1 for c in url if c.isdigit()) / len(url) if len(url) > 0 else 0)  # digit_ratio
        special_chars = sum(1 for c in url if not c.isalnum() and c not in '.-_/:?=&')
        features.append(special_chars)  # special_char_count
        
        # URL structure analysis
        features.append(1 if len(domain.split('.')) > 3 else 0)  # has_multiple_subdomains
        features.append(1 if any(char in domain for char in ['@', '%']) else 0)  # has_suspicious_chars
        
        # Phishing-like patterns
        legitimate_domains = ['google', 'facebook', 'amazon', 'microsoft', 'apple', 'paypal']
        features.append(1 if any(domain_name in domain and domain != f'{domain_name}.com' 
                               for domain_name in legitimate_domains) else 0)  # impersonating_legitimate
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features from URL {url}: {str(e)}")
        # Return default safe features if extraction fails
        return [0.0] * 16

# Load ML model (you'll need to train and save this)
model = None
try:
    if os.path.exists('phishing_model.pkl'):
        model = joblib.load('phishing_model.pkl')
        logger.info("ML model loaded successfully")
    else:
        logger.warning("ML model not found. Using rule-based detection.")
except Exception as e:
    logger.error(f"Error loading ML model: {str(e)}")
    model = None

def rule_based_prediction(url: str, features: List[float]) -> tuple[bool, float]:
    """Rule-based prediction when ML model is not available"""
    risk_score = 0
    
    # Check for suspicious patterns
    suspicious_patterns = [
        'bit.ly', 'tinyurl', 'secure', 'verify', 'suspended', 'urgent',
        'paypal', 'amazon', 'microsoft', 'google', 'apple', 'banking'
    ]
    
    url_lower = url.lower()
    for pattern in suspicious_patterns:
        if pattern in url_lower:
            risk_score += 0.2
    
    # Check URL length
    if len(url) > 75:
        risk_score += 0.3
    
    # Check for IP address
    if re.match(r'^\d+\.\d+\.\d+\.\d+', urlparse(url).netloc):
        risk_score += 0.4
    
    # Check for multiple subdomains
    domain = urlparse(url).netloc
    if domain.count('.') > 2:
        risk_score += 0.2
    
    # Check for suspicious characters
    if any(char in domain for char in ['-', '_', '@', '%']):
        risk_score += 0.1
    
    is_phishing = risk_score > 0.5
    confidence = min(0.95, max(0.60, risk_score + 0.3))
    
    return is_phishing, confidence

@app.post("/predict", response_model=PredictionResponse)
async def predict_url(request: URLRequest):
    """Predict whether a URL is phishing or safe"""
    try:
        url = request.url.strip()
        
        # Validate URL
        if not url:
            raise HTTPException(status_code=400, detail="URL cannot be empty")
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
        
        # Validate URL format
        try:
            parsed_url = urlparse(url)
            if not parsed_url.netloc:
                raise ValueError("Invalid URL format")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid URL format")
        
        # Extract features
        features = extract_features(url)
        
        # Make prediction
        if model:
            # Use trained ML model
            prediction_proba = model.predict_proba([features])[0]
            is_phishing = prediction_proba[1] > 0.5  # Assuming class 1 is phishing
            confidence = max(prediction_proba)
        else:
            # Use rule-based prediction
            is_phishing, confidence = rule_based_prediction(url, features)
        
        # Generate explanation and factors
        risk_factors = []
        safe_indicators = []
        
        if is_phishing:
            # Generate risk factors based on features
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            if len(url) > 75:
                risk_factors.append("URL is unusually long, which is common in phishing attacks")
            
            if re.match(r'^\d+\.\d+\.\d+\.\d+', domain):
                risk_factors.append("Uses IP address instead of domain name")
            
            shortening_services = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl']
            if any(service in domain for service in shortening_services):
                risk_factors.append("Uses URL shortening service to hide destination")
            
            suspicious_words = ['secure', 'verify', 'suspended', 'urgent', 'paypal', 'amazon']
            if any(word in url.lower() for word in suspicious_words):
                risk_factors.append("Contains suspicious keywords commonly used in phishing")
            
            if domain.count('.') > 2:
                risk_factors.append("Has multiple subdomains which may indicate deception")
            
            if not risk_factors:
                risk_factors.append("Multiple suspicious patterns detected by AI model")
            
            explanation = "Our AI model has identified several indicators suggesting this URL may be a phishing attempt designed to steal personal information or install malware."
            
        else:
            # Generate safe indicators
            parsed = urlparse(url)
            
            if parsed.scheme == 'https':
                safe_indicators.append("Uses HTTPS encryption")
            
            if len(url) < 50:
                safe_indicators.append("URL length is reasonable")
            
            if '.' in parsed.netloc and not re.match(r'^\d+\.\d+\.\d+\.\d+', parsed.netloc):
                safe_indicators.append("Uses proper domain name structure")
            
            suspicious_words = ['secure', 'verify', 'suspended', 'urgent']
            if not any(word in url.lower() for word in suspicious_words):
                safe_indicators.append("No suspicious keywords detected")
            
            if not safe_indicators:
                safe_indicators.append("No significant red flags detected by AI analysis")
            
            explanation = "The URL appears to be legitimate with no significant security concerns identified by our AI model."
        
        return PredictionResponse(
            is_phishing=is_phishing,
            confidence=round(confidence, 3),
            explanation=explanation,
            risk_factors=risk_factors,
            safe_indicators=safe_indicators
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for URL {request.url}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while analyzing the URL"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Phishing URL Detector API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)