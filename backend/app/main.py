from flask import Flask, request, jsonify
from summarizer import generate_summary
from detector import detect_fake_news
from explain import explain_prediction
from utils import preprocess_text
from flask_cors import CORS
import requests

NEWS_API_KEY ='3dc99c57fb8a6d468901891e9f631a96'  # TODO: Replace with your NewsAPI key

app = Flask(__name__)
CORS(app)

@app.route('/summarize_detect', methods=['POST'])
def summarize_and_detect():
    data = request.json
    input_data = data.get('text') or data.get('url')
    model_choice = data.get('summarizer_model', 'bart')
    article_metadata = data.get('article_metadata', {})

    print(f"DEBUG: Received data - text length: {len(input_data) if input_data else 0}")
    print(f"DEBUG: Article metadata: {article_metadata}")

    if not input_data:
        return jsonify({'error': 'No news text or URL provided'}), 400

    # Preprocess
    clean_text, tokens, tfidf_vec, sbert_vec, metadata = preprocess_text(input_data)
    print(f"DEBUG: Initial metadata from preprocess: {metadata}")

    # If we have article metadata, use it instead of the extracted metadata
    if article_metadata:
        print(f"DEBUG: Using article metadata: {article_metadata}")
        source = article_metadata.get('source')
        if isinstance(source, dict):
            source_name = source.get('name', 'Unknown Source')
        elif isinstance(source, str):
            source_name = source
        else:
            source_name = 'Unknown Source'

        metadata.update({
            'source': source_name,
            'publish_date': article_metadata.get('publishedAt', article_metadata.get('publish_date', 'unknown')),
            'title': article_metadata.get('title', ''),
            'author': article_metadata.get('author', 'Unknown Author'),
            'url': article_metadata.get('url', '')
        })
        print(f"DEBUG: Updated metadata: {metadata}")

    # Ensure all expected metadata fields are present
    for key, default in [
        ('source', 'Unknown Source'),
        ('publish_date', 'unknown'),
        ('title', ''),
        ('author', 'Unknown Author'),
        ('url', '')
    ]:
        if key not in metadata or metadata[key] is None:
            metadata[key] = default

    # Summarize
    summary = generate_summary(clean_text, model=model_choice)
    print(f"DEBUG: Summary generated: {len(summary)} characters")

    # Detect fake news
    label, score = detect_fake_news(tfidf_vec, sbert_vec, metadata, clean_text)
    print(f"DEBUG: Detection result - Label: {label}, Score: {score}")

    # Explain
    print(f"DEBUG: Calling explain_prediction...")
    explanation = explain_prediction(clean_text, label, tokens, metadata)
    print(f"DEBUG: Explanation received: {explanation}")

    # Ensure explanation fields are present
    explanation_fields = {
        'explanation': '',
        'top_words': [],
        'top_metadata': []
    }
    for key, default in explanation_fields.items():
        if key not in explanation or explanation[key] is None:
            explanation[key] = default

    return jsonify({
        'summary': summary or '',
        'credibility_label': label or '',
        'credibility_score': score if score is not None else 0,
        'explanation': explanation['explanation'],
        'top_words': explanation['top_words'],
        'top_metadata': explanation['top_metadata'],
        'metadata': metadata
    })

@app.route('/news', methods=['GET'])
def get_latest_news():
    url = 'https://gnews.io/api/v4/top-headlines'
    params = {
        'token': NEWS_API_KEY,      # GNews uses 'token' instead of 'apiKey'
        'lang': 'en',
        'country': 'in',            # For India
        'max': 10                   # Number of articles
    }
   
    response = requests.get(url, params=params)
   

    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch news', 'details': response.text}), 500

    try:
        data = response.json()
    except Exception as e:
        print(f"ERROR: Could not decode JSON: {e}")
        print(f"Response text: {response.text}")
        return jsonify({'error': 'Invalid response from GNews API', 'details': response.text}), 500

    if data.get('articles') and len(data['articles']) > 0:
        print(f"DEBUG: First article structure: {data['articles'][0]}")

    articles = [
        {
            'title': article.get('title', ''),
            'description': article.get('description', ''),
            'url': article.get('url', ''),
            'content': article.get('content', ''),
            'source': {
                'name': article.get('source', {}).get('name', 'Unknown Source'),
                'id': article.get('source', {}).get('url', 'unknown')
            },
            'publishedAt': article.get('publishedAt', ''),
            'author': article.get('author', 'Unknown Author'),
            # Add any other available fields from GNews here
            'image': article.get('image', ''),
            'category': article.get('category', ''),
        }
        for article in data.get('articles', [])
    ]
    return jsonify({'articles': articles})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
