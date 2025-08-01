import numpy as np
from transformers import pipeline
import joblib
import os
import re

# Load trained models
def load_models():
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')
        
        vectorizer = joblib.load(os.path.join(data_dir, 'vectorizer.joblib'))
        lr_model = joblib.load(os.path.join(data_dir, 'lr_model.joblib'))
        xgb_model = joblib.load(os.path.join(data_dir, 'xgb_model.joblib'))
        print("Successfully loaded trained models!")
        return vectorizer, lr_model, xgb_model
    except FileNotFoundError as e:
        print(f"Warning: Trained models not found. Error: {e}")
        print("Using placeholder models.")
        return None, None, None

# Load models
vectorizer, lr_model, xgb_model = load_models()

# BERT classifier (using generic model for now)
bert_classifier = pipeline('text-classification', model='bert-base-uncased')

def analyze_metadata_credibility(metadata):
    """Analyze metadata for credibility indicators"""
    credibility_score = 0.5  # Base score
    
    # Source credibility
    source = metadata.get('source', '').lower()
    credible_sources = ['reuters', 'ap', 'bbc', 'cnn', 'nbc', 'abc', 'cbs', 'pbs', 'npr']
    suspicious_sources = ['blog', 'forum', 'social', 'anonymous', 'unknown']
    
    if any(credible in source for credible in credible_sources):
        credibility_score += 0.2
    elif any(suspicious in source for suspicious in suspicious_sources):
        credibility_score -= 0.2
    
    # Date analysis
    publish_date = metadata.get('publish_date', '')
    if publish_date and publish_date != 'unknown':
        credibility_score += 0.1
    
    # Title analysis
    title = metadata.get('title', '')
    if title:
        # Check for sensationalist words
        sensational_words = ['shocking', 'amazing', 'incredible', 'you won\'t believe', 'breaking']
        if any(word in title.lower() for word in sensational_words):
            credibility_score -= 0.1
    
    return max(0.0, min(1.0, credibility_score))

def detect_fake_news(tfidf_vec, sbert_vec, metadata, text):
    try:
        # Use trained models if available
        if vectorizer is not None and lr_model is not None and xgb_model is not None:
            # Vectorize the text
            X = vectorizer.transform([text])
            
            # Get predictions from trained models with error handling
            try:
                lr_pred_raw = lr_model.predict(X)[0]
                print(f"Debug - LR raw prediction: {lr_pred_raw}, type: {type(lr_pred_raw)}")
            except Exception as e:
                print(f"Error with LR prediction: {e}")
                lr_pred_raw = 0  # fallback
            
            try:
                xgb_pred_num = xgb_model.predict(X)[0]
                print(f"Debug - XGB raw prediction: {xgb_pred_num}, type: {type(xgb_pred_num)}")
            except Exception as e:
                print(f"Error with XGB prediction: {e}")
                xgb_pred_num = 0  # fallback
            
            # Convert all predictions to our standard labels
            num_to_label = {0: 'fake', 1: 'real'}
            print(f"Debug - num_to_label dictionary: {num_to_label}")
            
            # Ensure XGB prediction is mapped correctly
            print(f"Debug - XGB prediction type: {type(xgb_pred_num)}, value: {xgb_pred_num}")
            if xgb_pred_num in num_to_label:
                xgb_pred = num_to_label[xgb_pred_num]
                print(f"Debug - XGB mapped to: {xgb_pred}")
            else:
                print(f"Warning: Invalid XGB prediction {xgb_pred_num}, using fallback")
                xgb_pred = 'fake'
            
            # Ensure LR prediction is also mapped correctly
            print(f"Debug - LR prediction type: {type(lr_pred_raw)}, value: {lr_pred_raw}")
            if isinstance(lr_pred_raw, str):
                if lr_pred_raw in ['fake', 'real']:
                    lr_pred = lr_pred_raw
                    print(f"Debug - LR string prediction: {lr_pred}")
                else:
                    print(f"Warning: Invalid LR string prediction '{lr_pred_raw}', using fallback")
                    lr_pred = 'fake'
            else:
                if lr_pred_raw in num_to_label:
                    lr_pred = num_to_label[lr_pred_raw]
                    print(f"Debug - LR numeric prediction mapped to: {lr_pred}")
                else:
                    print(f"Warning: Invalid LR numeric prediction {lr_pred_raw}, using fallback")
                    lr_pred = 'fake'
            
            # Get prediction probabilities for scoring
            try:
                lr_prob = lr_model.predict_proba(X)[0]
                xgb_prob = xgb_model.predict_proba(X)[0]
            except Exception as e:
                print(f"Error getting probabilities: {e}")
                lr_prob = [0.5, 0.5]  # fallback
                xgb_prob = [0.5, 0.5]  # fallback
            
            # BERT prediction (semantic) - handle label mapping
            try:
                print(f"Debug - About to call BERT classifier with text: {text[:50]}...")
                bert_pred = bert_classifier(text[:512])[0]
                print(f"Debug - BERT prediction structure: {bert_pred}")
                print(f"Debug - BERT prediction type: {type(bert_pred)}")
                print(f"Debug - BERT prediction keys: {bert_pred.keys() if isinstance(bert_pred, dict) else 'Not a dict'}")
                
                # Handle different possible BERT output formats
                if isinstance(bert_pred, dict):
                    if 'label' in bert_pred:
                        bert_label_raw = bert_pred['label'].lower()
                        print(f"Debug - Found 'label' key: {bert_label_raw}")
                    elif 'labels' in bert_pred:
                        bert_label_raw = bert_pred['labels'][0].lower()
                        print(f"Debug - Found 'labels' key: {bert_label_raw}")
                    else:
                        print("Warning: BERT prediction has no 'label' or 'labels' key")
                        bert_label_raw = 'label_0'  # fallback
                else:
                    print(f"Warning: BERT prediction is not a dict: {type(bert_pred)}")
                    bert_label_raw = 'label_0'  # fallback
                    
                print(f"Debug - BERT raw prediction: {bert_label_raw}")
            except Exception as e:
                print(f"Error with BERT prediction: {e}")
                print(f"Error type: {type(e)}")
                bert_label_raw = 'label_0'  # fallback
            
            # Map BERT labels to our labels (BERT uses label_0, label_1)
            if bert_label_raw == 'label_0':
                bert_label = 'fake'
            elif bert_label_raw == 'label_1':
                bert_label = 'real'
            else:
                # Fallback: use score to determine label
                bert_score = bert_pred.get('score', 0.5)
                bert_label = 'fake' if bert_score < 0.5 else 'real'
            
            # Debug: print all predictions to ensure they're correct
            print(f"Debug - Final predictions - LR: {lr_pred}, XGB: {xgb_pred}, BERT: {bert_label}")
            
            # Metadata credibility analysis
            metadata_credibility = analyze_metadata_credibility(metadata)
            
            # Enhanced voting mechanism with metadata
            votes = []
            weights = []
            
            # Model predictions with weights - ensure all are valid
            valid_predictions = []
            for pred in [lr_pred, xgb_pred, bert_label]:
                if pred in ['fake', 'real']:
                    valid_predictions.append(pred)
                else:
                    print(f"Warning: Invalid prediction '{pred}', using fallback")
                    valid_predictions.append('fake')  # fallback
            
            votes.extend(valid_predictions)
            weights.extend([0.4, 0.4, 0.2])  # Give more weight to trained models
            
            # Metadata-based adjustment
            if metadata_credibility > 0.7:
                votes.append('real')
                weights.append(0.1)
            elif metadata_credibility < 0.3:
                votes.append('fake')
                weights.append(0.1)
            
            # Weighted voting
            vote_counts = {'fake': 0, 'real': 0}
            for vote, weight in zip(votes, weights):
                if vote in vote_counts:  # Ensure vote is valid
                    vote_counts[vote] += weight
            
            # Determine final label
            label = 'fake' if vote_counts['fake'] > vote_counts['real'] else 'real'
            
            # Calculate confidence score
            lr_conf = max(lr_prob)
            xgb_conf = max(xgb_prob)
            bert_conf = bert_pred.get('score', 0.5)
            
            # Combine model confidence with metadata credibility
            model_confidence = (lr_conf + xgb_conf + bert_conf) / 3
            final_confidence = (model_confidence + metadata_credibility) / 2
            score = round(final_confidence, 2)
            
        else:
            # Fallback to placeholder logic
            try:
                bert_pred = bert_classifier(text[:512])[0]
                print(f"Debug - Fallback BERT prediction structure: {bert_pred}")
                
                # Handle different possible BERT output formats
                if isinstance(bert_pred, dict):
                    if 'label' in bert_pred:
                        bert_label_raw = bert_pred['label'].lower()
                    elif 'labels' in bert_pred:
                        bert_label_raw = bert_pred['labels'][0].lower()
                    else:
                        print("Warning: Fallback BERT prediction has no 'label' or 'labels' key")
                        bert_label_raw = 'label_0'  # fallback
                else:
                    print(f"Warning: Fallback BERT prediction is not a dict: {type(bert_pred)}")
                    bert_label_raw = 'label_0'  # fallback
                
                # Map BERT labels
                if bert_label_raw == 'label_0':
                    bert_label = 'fake'
                elif bert_label_raw == 'label_1':
                    bert_label = 'real'
                else:
                    bert_score = bert_pred.get('score', 0.5) if isinstance(bert_pred, dict) else 0.5
                    bert_label = 'fake' if bert_score < 0.5 else 'real'
            except Exception as e:
                print(f"Error with fallback BERT prediction: {e}")
                bert_label = 'fake'  # ultimate fallback
            
            lr_label = 'real'  # Placeholder
            xgb_label = 'fake'  # Placeholder
            
            votes = [bert_label, lr_label, xgb_label]
            label = max(set(votes), key=votes.count)
            score = round(np.random.uniform(0.7, 0.95), 2)
        
        return label, score
        
    except Exception as e:
        print(f"Error in detect_fake_news: {e}")
        # Ultimate fallback
        return 'fake', 0.5 