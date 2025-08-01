import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import pandas as pd

# Initialize vectorizers
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
tfidf_vectorizer = TfidfVectorizer()

# For demo, fit TF-IDF on a small corpus (should be fit on a large corpus in production)
tfidf_vectorizer.fit(["news article example", "another example text"])

def extract_metadata_from_url(url):
    try:
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        title = soup.title.string if soup.title else ''
        source = urlparse(url).netloc
        publish_date = ''
        for meta in soup.find_all('meta'):
            if 'date' in str(meta).lower():
                publish_date = meta.get('content', '')
                break
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text, {'source': source, 'publish_date': publish_date, 'title': title}
    except Exception:
        return '', {'source': '', 'publish_date': '', 'title': ''}

def preprocess_text(text_or_url):
    # If input looks like a URL, fetch and extract
    if re.match(r'^https?://', text_or_url):
        text, metadata = extract_metadata_from_url(text_or_url)
    else:
        text = text_or_url
        metadata = {'source': 'unknown', 'publish_date': 'unknown', 'title': ''}
    # Lowercase and remove punctuation
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    # Tokenize
    tokens = clean.split()
    # TF-IDF vector
    tfidf_vec = tfidf_vectorizer.transform([' '.join(tokens)]).toarray()[0]
    # SBERT vector
    sbert_vec = sbert_model.encode([' '.join(tokens)])[0]
    return clean, tokens, tfidf_vec, sbert_vec, metadata

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