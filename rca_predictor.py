#!/usr/bin/env python
"""
AutoRCA ML Predictor

This script contains ML models to predict Root Cause Analysis (RCA) for issues
without existing RCA information. It trains on issues with existing RCA (80%)
and predicts causes for the remaining 20% of issues with supporting evidence.

Author: Kedar Haldankar
"""

import os
import csv
import json
import logging
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


# Initialize logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('rca_predictor')

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('tokenizers/punkt/PY3/english.pickle', quiet=True)
    # Create the required directory if it doesn't exist
    import os
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    if not os.path.exists(os.path.join(nltk_data_dir, 'tokenizers')):
        os.makedirs(os.path.join(nltk_data_dir, 'tokenizers'))
    if not os.path.exists(os.path.join(nltk_data_dir, 'tokenizers/punkt_tab')):
        os.makedirs(os.path.join(nltk_data_dir, 'tokenizers/punkt_tab'))
    if not os.path.exists(os.path.join(nltk_data_dir, 'tokenizers/punkt_tab/english')):
        os.makedirs(os.path.join(nltk_data_dir, 'tokenizers/punkt_tab/english'))
except Exception as e:
    logger.warning(f"Error downloading NLTK resources: {e}")
    pass  # Continue even if download fails

# Define paths
JIRA_ISSUES_DIR = 'jira_issues'
LOGS_DIR = 'logs'
TRAIN_ISSUES_PATH = os.path.join(JIRA_ISSUES_DIR, 'train_issues.csv')
TEST_ISSUES_PATH = os.path.join(JIRA_ISSUES_DIR, 'test_issues.csv')
FULL_ISSUES_PATH = os.path.join(JIRA_ISSUES_DIR, 'jira_issues.csv')
OUTPUT_PREDICTIONS_PATH = os.path.join(JIRA_ISSUES_DIR, 'predicted_issues.csv')

# Define root cause categories
ROOT_CAUSE_CATEGORIES = [
    "Code Bug",
    "Performance Bottleneck", 
    "Configuration Error",
    "Database Issue",
    "Network Issue",
    "External Dependency Failure",
    "Resource Limitation",
    "Security Issue", 
    "Data Corruption",
    "Race Condition",
    "Memory Leak"
]


class LogDataExtractor:
    """Class to extract relevant data from log files"""
    
    def __init__(self, logs_dir=LOGS_DIR):
        """Initialize with the logs directory"""
        self.logs_dir = logs_dir
        self.logs_data = {}
        self.load_logs()
        
    def load_logs(self):
        """Load all logs from CSV files"""
        log_files = {
            'datadog': os.path.join(self.logs_dir, 'datadog_logs.csv'),
            'backend': os.path.join(self.logs_dir, 'backend_logs.csv'),
            'analytics': os.path.join(self.logs_dir, 'analytics_logs.csv')
        }
        
        for source, filepath in log_files.items():
            if os.path.exists(filepath):
                try:
                    self.logs_data[source] = pd.read_csv(filepath)
                    logger.info(f"Loaded {len(self.logs_data[source])} logs from {source}")
                except Exception as e:
                    logger.error(f"Error loading {source} logs: {e}")
                    self.logs_data[source] = pd.DataFrame()
            else:
                logger.warning(f"Log file not found: {filepath}")
                self.logs_data[source] = pd.DataFrame()
    
    def find_relevant_logs(self, issue_data, max_logs=5):
        """Find logs relevant to a specific issue based on service, component, and timestamp"""
        relevant_logs = []
        
        # Extract issue information
        service = issue_data.get('service', '')
        component = issue_data.get('component', '')
        created_at = issue_data.get('created', '')
        
        # Search criteria (expand as needed)
        search_terms = []
        if service:
            search_terms.append(service.lower())
        if component:
            search_terms.append(component.lower())
            
        # Extract error type if available
        description = issue_data.get('description', '')
        if 'error' in description.lower():
            error_lines = [line for line in description.split('\n') if 'error' in line.lower()]
            if error_lines:
                error_text = error_lines[0]
                # Extract potential error message
                if ':' in error_text:
                    error_type = error_text.split(':', 1)[1].strip()
                    search_terms.append(error_type.lower())
        
        # Go through each log source
        for source, logs_df in self.logs_data.items():
            if logs_df.empty:
                continue
                
            # Filter by service if available
            filtered_logs = logs_df
            if service and 'service' in logs_df.columns:
                filtered_logs = filtered_logs[
                    filtered_logs['service'].str.lower().str.contains(service.lower(), na=False)
                ]
            
            # If still too many logs, filter by timestamp if available
            if len(filtered_logs) > 100 and created_at and 'timestamp' in filtered_logs.columns:
                # Consider logs from 24 hours before issue creation
                created_datetime = pd.to_datetime(created_at)
                filtered_logs = filtered_logs[
                    pd.to_datetime(filtered_logs['timestamp']) <= created_datetime
                ]
                # Keep only most recent logs
                filtered_logs = filtered_logs.sort_values(
                    by='timestamp', ascending=False
                ).head(100)
            
            # Search for relevant terms in message field
            relevance_scores = []
            for idx, log in filtered_logs.iterrows():
                score = 0
                log_message = str(log.get('message', '')) + str(log.get('error_message', ''))
                for term in search_terms:
                    if term in log_message.lower():
                        score += 1
                
                # Higher score for error logs
                if 'level' in log and log['level'].lower() in ['error', 'critical', 'fatal']:
                    score += 2
                    
                if score > 0:
                    relevance_scores.append((score, idx))
            
            # Get top logs by relevance score
            for score, idx in sorted(relevance_scores, reverse=True)[:max_logs]:
                log_entry = filtered_logs.iloc[idx].to_dict()
                log_entry['source'] = source
                log_entry['relevance_score'] = score
                relevant_logs.append(log_entry)
        
        # Sort by relevance score and return top logs
        relevant_logs = sorted(relevant_logs, key=lambda x: x['relevance_score'], reverse=True)
        return relevant_logs[:max_logs]


class TextPreprocessor:
    """Class to preprocess text data for ML models"""
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text):
        """Preprocess a text string"""
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Simple tokenization without relying on punkt_tab
        try:
            # Try to use nltk word_tokenize
            tokens = word_tokenize(text)
        except Exception as e:
            # Fallback to simple space-based tokenization if word_tokenize fails
            logger.warning(f"Falling back to simple tokenization: {e}")
            tokens = text.split()
        
        # Remove stop words and non-alphabetic tokens
        tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        
        # Join tokens back to string
        return " ".join(tokens)


class RCAPredictor:
    """Class for predicting Root Cause Analysis for issues"""
    
    def __init__(self):
        """Initialize the RCA predictor"""
        self.preprocessor = TextPreprocessor()
        self.log_extractor = LogDataExtractor()
        self.model = None
        self.vectorizer = None
        self.label_mapping = None
        self.feature_importance = None
    
    def prepare_training_data(self, issues_df):
        """Prepare training data from issues with RCA"""
        # Filter issues with RCA
        train_issues = issues_df[issues_df['has_rca'] == True].copy()
        
        if train_issues.empty:
            logger.error("No training issues with RCA found!")
            return None, None
            
        # Extract text features from description and summary
        train_issues['processed_text'] = (
            train_issues['summary'].fillna('') + ' ' + 
            train_issues['description'].fillna('')
        ).apply(self.preprocessor.preprocess)
        
        # Prepare X (features) and y (labels)
        X = train_issues['processed_text']
        
        # Extract root cause from RCA
        def extract_root_cause(rca_json):
            try:
                if isinstance(rca_json, str):
                    rca = json.loads(rca_json)
                else:
                    rca = rca_json
                return rca.get('root_cause_category', '')
            except:
                return ''
        
        train_issues['root_cause'] = train_issues['rca'].apply(extract_root_cause)
        
        # One-hot encode the root causes
        y = pd.get_dummies(train_issues['root_cause'])
        
        # Ensure all categories are present
        for category in ROOT_CAUSE_CATEGORIES:
            if category not in y.columns:
                y[category] = 0
        
        # Keep only known categories
        y = y[[c for c in y.columns if c in ROOT_CAUSE_CATEGORIES]]
        
        # Store label mapping
        self.label_mapping = {i: label for i, label in enumerate(y.columns)}
        
        logger.info(f"Prepared training data with {len(X)} samples and {len(y.columns)} classes")
        return X, y
    
    def train(self, issues_df):
        """Train the RCA prediction model"""
        X, y = self.prepare_training_data(issues_df)
        
        if X is None or y is None:
            logger.error("Training failed: Could not prepare training data")
            return False
        
        # Create pipeline with vectorizer and classifier
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=5000, 
                ngram_range=(1, 2),
                stop_words='english'
            )),
            ('classifier', MultiOutputClassifier(
                RandomForestClassifier(
                    n_estimators=100, 
                    random_state=42,
                    class_weight='balanced'
                )
            ))
        ])
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        logger.info("Training RCA prediction model...")
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate on validation set
        logger.info("Evaluating model on validation set...")
        y_pred = self.pipeline.predict(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        logger.info(f"Validation accuracy: {accuracy:.4f}")
        logger.info(f"Validation F1 score: {f1:.4f}")
        
        # Store vectorizer for feature importance
        self.vectorizer = self.pipeline.named_steps['vectorizer']
        classifier = self.pipeline.named_steps['classifier']
        
        # Extract feature importance
        feature_names = self.vectorizer.get_feature_names_out()
        feature_importance = []
        
        for i, estimator in enumerate(classifier.estimators_):
            if hasattr(estimator, 'feature_importances_'):
                label = self.label_mapping.get(i, f"Label_{i}")
                importances = estimator.feature_importances_
                sorted_idx = np.argsort(importances)[::-1][:10]  # Top 10 features
                
                for idx in sorted_idx:
                    if idx < len(feature_names):
                        feature_importance.append({
                            'label': label,
                            'feature': feature_names[idx],
                            'importance': float(importances[idx])
                        })
        
        self.feature_importance = feature_importance
        logger.info(f"Model training completed with {len(feature_importance)} important features extracted")
        return True
    
    def predict(self, issues_df):
        """Predict RCA for issues without existing RCA"""
        if self.pipeline is None:
            logger.error("Model not trained yet!")
            return issues_df
            
        # Filter issues without RCA
        predict_issues = issues_df[issues_df['has_rca'] == False].copy()
        
        if predict_issues.empty:
            logger.info("No issues without RCA found for prediction")
            return issues_df
            
        logger.info(f"Predicting RCA for {len(predict_issues)} issues")
        
        # Prepare features
        predict_issues['processed_text'] = (
            predict_issues['summary'].fillna('') + ' ' + 
            predict_issues['description'].fillna('')
        ).apply(self.preprocessor.preprocess)
        
        X_pred = predict_issues['processed_text']
        
        # Make predictions
        y_pred_prob = self.pipeline.predict_proba(X_pred)
        
        # Process predictions
        predicted_issues = []
        
        for idx, issue in predict_issues.iterrows():
            issue_dict = issue.to_dict()
            
            # Get probabilities for each class
            issue_probs = []
            max_prob = 0
            predicted_class = None
            
            for i, estimator in enumerate(self.pipeline.named_steps['classifier'].estimators_):
                label = self.label_mapping.get(i, f"Label_{i}")
                prob = y_pred_prob[i][idx][1]  # Probability of positive class
                issue_probs.append((label, prob))
                
                if prob > max_prob:
                    max_prob = prob
                    predicted_class = label
            
            # Sort by probability
            issue_probs = sorted(issue_probs, key=lambda x: x[1], reverse=True)
            
            # Find supporting evidence from logs
            supporting_logs = self.log_extractor.find_relevant_logs(issue_dict)
            
            # Extract key evidence
            evidence = []
            for log in supporting_logs:
                if 'message' in log:
                    message = log.get('message', '')
                    if len(message) > 200:
                        message = message[:200] + "..."
                    
                    evidence.append({
                        'source': log.get('source', 'unknown'),
                        'level': log.get('level', ''),
                        'message': message,
                        'relevance': log.get('relevance_score', 0)
                    })
            
            # Find feature importance for the prediction
            important_features = []
            if self.feature_importance:
                for item in self.feature_importance:
                    if item['label'] == predicted_class:
                        feature = item['feature']
                        if feature in issue_dict['processed_text']:
                            important_features.append({
                                'feature': feature,
                                'importance': item['importance']
                            })
            
            # Sort by importance
            important_features = sorted(
                important_features, 
                key=lambda x: x['importance'], 
                reverse=True
            )[:5]
            
            # Create RCA prediction JSON
            rca_prediction = {
                'predicted': True,
                'root_cause_category': predicted_class,
                'confidence': float(max_prob),
                'alternative_causes': [
                    {'category': label, 'probability': float(prob)} 
                    for label, prob in issue_probs[:3] if prob > 0.05
                ],
                'supporting_evidence': evidence,
                'important_features': important_features
            }
            
            # Update issue with prediction
            issue_dict['rca'] = json.dumps(rca_prediction)
            issue_dict['predicted_rca'] = True
            issue_dict['rca_confidence'] = float(max_prob)
            
            predicted_issues.append(issue_dict)
        
        # Create new dataframe with predictions
        predicted_df = pd.DataFrame(predicted_issues)
        
        # Merge with original dataframe (keep issues with RCA)
        result_df = pd.concat([
            issues_df[issues_df['has_rca'] == True],
            predicted_df
        ])
        
        logger.info(f"RCA prediction completed for {len(predicted_issues)} issues")
        return result_df


def load_jira_issues(filepath):
    """Load Jira issues from CSV file"""
    if not os.path.exists(filepath):
        logger.error(f"Issues file not found: {filepath}")
        return pd.DataFrame()
        
    try:
        issues_df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(issues_df)} issues from {filepath}")
        
        # Convert string 'True'/'False' to boolean
        if 'has_rca' in issues_df.columns:
            issues_df['has_rca'] = issues_df['has_rca'].astype(str).map({'True': True, 'False': False})
        
        return issues_df
    except Exception as e:
        logger.error(f"Error loading issues: {e}")
        return pd.DataFrame()


def save_predicted_issues(issues_df, filepath):
    """Save predicted issues to CSV file"""
    try:
        issues_df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(issues_df)} issues with predictions to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving predicted issues: {e}")
        return False


def perform_rca_prediction(input_file=None, output_file=None):
    """Perform RCA prediction on issues without existing RCA"""
    # Determine input file
    if input_file is None or not os.path.exists(input_file):
        # Try train/test split files first
        if os.path.exists(TRAIN_ISSUES_PATH) and os.path.exists(TEST_ISSUES_PATH):
            train_df = load_jira_issues(TRAIN_ISSUES_PATH)
            test_df = load_jira_issues(TEST_ISSUES_PATH)
            
            # Combine for training and prediction
            all_issues = pd.concat([train_df, test_df])
        else:
            # Fall back to full issues file
            all_issues = load_jira_issues(FULL_ISSUES_PATH)
    else:
        all_issues = load_jira_issues(input_file)
    
    if all_issues.empty:
        logger.error("No issues found for prediction!")
        return None
    
    # Initialize and train predictor
    predictor = RCAPredictor()
    success = predictor.train(all_issues)
    
    if not success:
        logger.error("Failed to train RCA predictor!")
        return None
    
    # Make predictions
    predicted_issues = predictor.predict(all_issues)
    
    # Save predictions
    if output_file is None:
        output_file = OUTPUT_PREDICTIONS_PATH
        
    save_predicted_issues(predicted_issues, output_file)
    
    return predicted_issues


def main():
    """Main function to run RCA prediction"""
    logger.info("Starting RCA prediction process")
    
    # Check if Jira issues exist
    if not os.path.exists(JIRA_ISSUES_DIR):
        logger.error(f"Jira issues directory not found: {JIRA_ISSUES_DIR}")
        return
    
    # Perform prediction
    predicted_issues = perform_rca_prediction()
    
    if predicted_issues is not None:
        # Calculate statistics
        total_issues = len(predicted_issues)
        issues_with_rca = sum(predicted_issues['has_rca'] == True)
        issues_predicted = sum(predicted_issues.get('predicted_rca', False))
        
        logger.info("RCA prediction completed successfully!")
        logger.info(f"Total issues: {total_issues}")
        logger.info(f"Issues with existing RCA: {issues_with_rca}")
        logger.info(f"Issues with predicted RCA: {issues_predicted}")
    else:
        logger.error("RCA prediction failed!")


if __name__ == "__main__":
    main()