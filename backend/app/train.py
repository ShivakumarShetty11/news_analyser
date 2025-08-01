import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

def load_liar_dataset(path):
    # LIAR dataset columns: [ID, label, statement, ...]
    df = pd.read_csv(path, sep='\t', header=None)
    # Use column 2 for label, column 3 for statement
    label_map = {
        'true': 'real',
        'mostly-true': 'real',
        'half-true': 'real',
        'barely-true': 'fake',
        'false': 'fake',
        'pants-fire': 'fake'
    }
    texts = df[2].astype(str).tolist()
    labels = df[1].map(label_map).tolist()
    return texts, labels

def train_models():
    print("Loading LIAR dataset...")
    
    # Load training data
    train_texts, train_labels = load_liar_dataset('data/train.tsv')
    valid_texts, valid_labels = load_liar_dataset('data/valid.tsv')
    test_texts, test_labels = load_liar_dataset('data/test.tsv')
    
    print(f"Training set: {len(train_texts)} samples")
    print(f"Validation set: {len(valid_texts)} samples")
    print(f"Test set: {len(test_texts)} samples")
    
    # Convert labels to numeric for XGBoost
    label_to_num = {'fake': 0, 'real': 1}
    train_labels_num = [label_to_num[label] for label in train_labels]
    valid_labels_num = [label_to_num[label] for label in valid_labels]
    test_labels_num = [label_to_num[label] for label in test_labels]
    
    # Create and fit TF-IDF vectorizer on training data
    print("Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train = vectorizer.fit_transform(train_texts)
    X_valid = vectorizer.transform(valid_texts)
    X_test = vectorizer.transform(test_texts)
    
    print(f"TF-IDF features: {X_train.shape[1]}")
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, train_labels)
    
    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = XGBClassifier(random_state=42, n_estimators=100)
    xgb_model.fit(X_train, train_labels_num)
    
    # Evaluate models
    print("\n=== Model Evaluation ===")
    
    # Logistic Regression evaluation
    lr_train_pred = lr_model.predict(X_train)
    lr_valid_pred = lr_model.predict(X_valid)
    lr_test_pred = lr_model.predict(X_test)
    
    print(f"Logistic Regression - Train Accuracy: {accuracy_score(train_labels, lr_train_pred):.4f}")
    print(f"Logistic Regression - Validation Accuracy: {accuracy_score(valid_labels, lr_valid_pred):.4f}")
    print(f"Logistic Regression - Test Accuracy: {accuracy_score(test_labels, lr_test_pred):.4f}")
    
    # XGBoost evaluation
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_valid_pred = xgb_model.predict(X_valid)
    xgb_test_pred = xgb_model.predict(X_test)
    
    # Convert numeric predictions back to labels
    num_to_label = {0: 'fake', 1: 'real'}
    xgb_train_pred_labels = [num_to_label[pred] for pred in xgb_train_pred]
    xgb_valid_pred_labels = [num_to_label[pred] for pred in xgb_valid_pred]
    xgb_test_pred_labels = [num_to_label[pred] for pred in xgb_test_pred]
    
    print(f"XGBoost - Train Accuracy: {accuracy_score(train_labels, xgb_train_pred_labels):.4f}")
    print(f"XGBoost - Validation Accuracy: {accuracy_score(valid_labels, xgb_valid_pred_labels):.4f}")
    print(f"XGBoost - Test Accuracy: {accuracy_score(test_labels, xgb_test_pred_labels):.4f}")
    
    # Save models and vectorizer
    print("\nSaving models...")
    joblib.dump(vectorizer, 'data/vectorizer.joblib')
    joblib.dump(lr_model, 'data/lr_model.joblib')
    joblib.dump(xgb_model, 'data/xgb_model.joblib')
    
    print("Training complete! Models saved to data/ directory.")
    print("\nDetailed Test Results:")
    print("Logistic Regression:")
    print(classification_report(test_labels, lr_test_pred))
    print("XGBoost:")
    print(classification_report(test_labels, xgb_test_pred_labels))

if __name__ == '__main__':
    train_models() 