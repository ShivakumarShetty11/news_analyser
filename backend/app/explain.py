import shap
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re

def explain_prediction(text, label, tokens, metadata):
    print(f"DEBUG: Starting explain_prediction for label: {label}")
    print(f"DEBUG: Text length: {len(text)}")
    print(f"DEBUG: Metadata: {metadata}")
    
    # Always use TF-IDF analysis as primary method (more reliable than SHAP)
    top_words = perform_tfidf_analysis(text)
    print(f"DEBUG: TF-IDF analysis found {len(top_words)} words")
    
    # Analyze metadata factors
    metadata_importance = []
    metadata_analysis = []
    
    # Source credibility analysis
    source = metadata.get('source', '').lower()
    print(f"DEBUG: Analyzing source: '{source}'")
    
    if source and source != 'unknown' and source != 'unknown source':
        credible_sources = ['reuters', 'ap', 'bbc', 'cnn', 'nbc', 'abc', 'cbs', 'pbs', 'npr', 'associated press', 'the new york times', 'washington post', 'wall street journal', 'axios', 'politico', 'bloomberg', 'forbes', 'time', 'newsweek']
        suspicious_sources = ['blog', 'forum', 'social', 'anonymous', 'unknown', 'rumor', 'conspiracy']
        
        if any(credible in source for credible in credible_sources):
            metadata_importance.append(f"Credible source: {metadata['source']}")
            metadata_analysis.append("High credibility due to reputable news source")
        elif any(suspicious in source for suspicious in suspicious_sources):
            metadata_importance.append(f"Suspicious source: {metadata['source']}")
            metadata_analysis.append("Low credibility due to unreliable source")
        else:
            metadata_importance.append(f"Source: {metadata['source']}")
            metadata_analysis.append("Moderate credibility source")
    else:
        metadata_importance.append("Source: Unknown")
        metadata_analysis.append("Unable to verify source credibility")
    
    # Date analysis
    publish_date = metadata.get('publish_date', '')
    if publish_date and publish_date != 'unknown':
        metadata_importance.append(f"Published: {publish_date}")
        metadata_analysis.append("Article has verifiable publication date")
    else:
        metadata_analysis.append("No publication date available")
    
    # Title analysis
    title = metadata.get('title', '')
    if title:
        sensational_words = ['shocking', 'amazing', 'incredible', 'you won\'t believe', 'breaking', 'exclusive', 'urgent', 'must see', 'viral']
        if any(word in title.lower() for word in sensational_words):
            metadata_importance.append("Sensationalist title detected")
            metadata_analysis.append("Title contains sensationalist language")
    
    # Content analysis
    content_analysis = analyze_content_credibility(text)
    metadata_analysis.extend(content_analysis)
    
    print(f"DEBUG: Final analysis - {len(top_words)} words, {len(metadata_analysis)} metadata factors")
    
    # Create comprehensive explanation
    if label == 'fake':
        explanation_text = f"This article was classified as FAKE NEWS with {len(top_words)} influential factors. "
        if top_words:
            explanation_text += f"The most significant words contributing to this classification were: {', '.join(top_words[:5])}. "
        
        if metadata_analysis:
            explanation_text += f"Analysis: {'; '.join(metadata_analysis)}. "
        
        explanation_text += "The combination of suspicious language patterns and metadata factors suggests this content may be misleading or false."
    else:
        explanation_text = f"This article was classified as REAL NEWS with {len(top_words)} influential factors. "
        if top_words:
            explanation_text += f"The most significant words contributing to this classification were: {', '.join(top_words[:5])}. "
        
        if metadata_analysis:
            explanation_text += f"Analysis: {'; '.join(metadata_analysis)}. "
        
        explanation_text += "The combination of factual language patterns and credible metadata suggests this content is likely accurate."
    
    return {
        'top_words': top_words[:10],
        'top_metadata': metadata_importance,
        'explanation': explanation_text,
        'shap_values': [],
        'metadata_analysis': metadata_analysis
    }

def perform_tfidf_analysis(text):
    """Perform TF-IDF analysis to find important words"""
    try:
        # Simple word frequency analysis with filtering
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs'}
        
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in stop_words:  # Only consider words longer than 3 characters
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words by frequency
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [f"{word} ({freq})" for word, freq in top_words]
    except Exception as e:
        print(f"DEBUG: TF-IDF analysis failed: {e}")
        return []

def analyze_content_credibility(text):
    """Analyze content for credibility indicators"""
    analysis = []
    text_lower = text.lower()
    
    # Check for sensationalist language
    sensational_words = ['amazing', 'incredible', 'shocking', 'you won\'t believe', 'viral', 'breaking', 'exclusive']
    if any(word in text_lower for word in sensational_words):
        analysis.append("Content contains sensationalist language")
    
    # Check for excessive punctuation
    if text.count('!') > text.count('.') * 0.3:
        analysis.append("Excessive use of exclamation marks detected")
    
    # Check for all caps
    if len(re.findall(r'\b[A-Z]{3,}\b', text)) > 3:
        analysis.append("Excessive use of capital letters detected")
    
    # Check for balanced reporting indicators
    balanced_indicators = ['however', 'although', 'nevertheless', 'on the other hand', 'according to', 'official', 'government', 'authority']
    if any(indicator in text_lower for indicator in balanced_indicators):
        analysis.append("Content shows balanced reporting indicators")
    
    # Check for factual language
    factual_indicators = ['according to', 'official', 'government', 'authority', 'report', 'study', 'research', 'data', 'statistics']
    if any(indicator in text_lower for indicator in factual_indicators):
        analysis.append("Content contains factual reporting indicators")
    
    return analysis

def create_basic_explanation(text, label, tokens, metadata):
    """Create a basic explanation when all else fails"""
    top_words = tokens[:5] if tokens else []
    top_metadata = [metadata.get('source', ''), metadata.get('publish_date', '')]
    
    if label == 'fake':
        explanation = f"This article was classified as FAKE NEWS. Key words: {', '.join(top_words)}. "
        explanation += f"Metadata factors: {', '.join([str(m) for m in top_metadata if m])}. "
        explanation += "The analysis suggests this content may be misleading."
    else:
        explanation = f"This article was classified as REAL NEWS. Key words: {', '.join(top_words)}. "
        explanation += f"Metadata factors: {', '.join([str(m) for m in top_metadata if m])}. "
        explanation += "The analysis suggests this content is likely accurate."
    
    return {
        'top_words': top_words,
        'top_metadata': top_metadata,
        'explanation': explanation,
        'shap_values': [],
        'metadata_analysis': []
    } 