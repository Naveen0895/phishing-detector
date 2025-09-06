"""
Machine Learning Model Training Script for Phishing URL Detection

This script trains various ML models on phishing URL datasets and saves the best performing model.

Usage:
1. Prepare your dataset as CSV with columns: 'url', 'is_phishing' (0 for safe, 1 for phishing)
2. Update the dataset path in the script
3. Run: python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from main import extract_features
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data(dataset_path: str):
    """Load and prepare the phishing dataset"""
    try:
        # Load dataset
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset with {len(df)} samples")
        
        # Ensure required columns exist
        if 'url' not in df.columns or 'is_phishing' not in df.columns:
            raise ValueError("Dataset must contain 'url' and 'is_phishing' columns")
        
        # Remove any missing values
        df = df.dropna(subset=['url', 'is_phishing'])
        logger.info(f"After removing NaN values: {len(df)} samples")
        
        # Extract features for each URL
        logger.info("Extracting features from URLs...")
        X = []
        y = []
        
        for idx, row in df.iterrows():
            try:
                features = extract_features(row['url'])
                X.append(features)
                y.append(int(row['is_phishing']))
                
                if idx % 1000 == 0:
                    logger.info(f"Processed {idx} URLs...")
                    
            except Exception as e:
                logger.warning(f"Error processing URL at index {idx}: {str(e)}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Feature extraction complete. Shape: {X.shape}")
        logger.info(f"Class distribution - Safe: {sum(y == 0)}, Phishing: {sum(y == 1)}")
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def train_models(X, y):
    """Train and evaluate different ML models"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"\nTraining {name}...")
        
        # Train model
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        if name == 'Logistic Regression':
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        logger.info(f"{name} Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    
    logger.info(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")
    
    return best_model, scaler, results

def hyperparameter_tuning(X, y, model_type='rf'):
    """Perform hyperparameter tuning for the best model"""
    logger.info("Performing hyperparameter tuning...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    if model_type == 'rf':
        # Random Forest hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
    else:
        # Logistic Regression hyperparameter tuning
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(
            lr, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Best parameters
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Test set evaluation
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Test set accuracy: {test_accuracy:.4f}")
    
    return best_model, (scaler if model_type == 'lr' else None)

def create_sample_dataset():
    """Create a sample dataset for testing"""
    logger.info("Creating sample dataset...")
    
    # Sample URLs (mix of safe and phishing)
    safe_urls = [
        'https://www.google.com',
        'https://www.github.com',
        'https://stackoverflow.com',
        'https://www.wikipedia.org',
        'https://www.amazon.com',
        'https://www.microsoft.com',
        'https://www.apple.com',
        'https://www.facebook.com',
        'https://www.twitter.com',
        'https://www.linkedin.com'
    ]
    
    phishing_urls = [
        'http://secure-paypal-verification.com',
        'https://amazon-security-update.net',
        'http://microsoft-account-suspended.org',
        'https://apple-id-verify-now.com',
        'http://facebook-security-alert.net',
        'https://google-account-recovery.org',
        'http://bit.ly/suspicious-link',
        'https://urgent-banking-update.com',
        'http://click-here-now-limited-time.net',
        'https://verify-account-immediately.org'
    ]
    
    # Create DataFrame
    urls = safe_urls + phishing_urls
    labels = [0] * len(safe_urls) + [1] * len(phishing_urls)
    
    df = pd.DataFrame({
        'url': urls,
        'is_phishing': labels
    })
    
    # Save sample dataset
    df.to_csv('sample_phishing_dataset.csv', index=False)
    logger.info("Sample dataset created: sample_phishing_dataset.csv")
    
    return 'sample_phishing_dataset.csv'

def main():
    """Main training pipeline"""
    # Dataset path - update this with your actual dataset
    dataset_path = 'phishing_dataset.csv'  # Update with your dataset path
    
    # Check if dataset exists, if not create sample
    try:
        df_test = pd.read_csv(dataset_path)
        logger.info(f"Using existing dataset: {dataset_path}")
    except FileNotFoundError:
        logger.warning(f"Dataset {dataset_path} not found. Creating sample dataset...")
        dataset_path = create_sample_dataset()
    
    try:
        # Load and prepare data
        X, y = load_and_prepare_data(dataset_path)
        
        # Train basic models
        best_model, scaler, results = train_models(X, y)
        
        # Perform hyperparameter tuning on best model
        if isinstance(best_model, RandomForestClassifier):
            tuned_model, tuned_scaler = hyperparameter_tuning(X, y, 'rf')
        else:
            tuned_model, tuned_scaler = hyperparameter_tuning(X, y, 'lr')
        
        # Save the best model
        joblib.dump(tuned_model, 'phishing_model.pkl')
        if tuned_scaler:
            joblib.dump(tuned_scaler, 'feature_scaler.pkl')
        
        # Save feature names for reference
        feature_names = [
            'url_length', 'domain_length', 'path_length', 'query_length',
            'has_ip', 'has_shortening', 'suspicious_words_count',
            'subdomain_count', 'dash_count', 'has_https', 'underscore_count',
            'digit_ratio', 'special_char_count', 'has_multiple_subdomains',
            'has_suspicious_chars', 'impersonating_legitimate'
        ]
        joblib.dump(feature_names, 'feature_names.pkl')
        
        logger.info("Model training completed successfully!")
        logger.info("Files saved:")
        logger.info("- phishing_model.pkl (trained model)")
        if tuned_scaler:
            logger.info("- feature_scaler.pkl (feature scaler)")
        logger.info("- feature_names.pkl (feature names)")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()