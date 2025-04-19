import os
import re
import nltk
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

LYRICS_DIR = 'lyrics'
SAVE_DIR = 'saved_models'
os.makedirs(SAVE_DIR, exist_ok=True)

TFIDF_VECTORIZER_PATH = os.path.join(SAVE_DIR, 'tfidf_vectorizer.pkl')
TFIDF_MATRIX_PATH = os.path.join(SAVE_DIR, 'tfidf_matrix.pkl')
BERT_MODEL_PATH = os.path.join(SAVE_DIR, 'bert_model')
BERT_EMBEDDINGS_PATH = os.path.join(SAVE_DIR, 'bert_embeddings.npy')

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    return ' '.join(words)

def load_lyrics():
    data = []
    for filename in os.listdir(LYRICS_DIR):
        if filename.endswith('.txt'):
            path = os.path.join(LYRICS_DIR, filename)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_lyrics = f.read()
                cleaned_lyrics = preprocess(raw_lyrics)

                filename_no_ext = filename[:-4]  # Remove ".txt"
                title = filename_no_ext.split('_')[0].strip()  # Extract before '_'

                data.append({
                    'title': title,
                    'lyrics': cleaned_lyrics
                })
    return pd.DataFrame(data)

df = load_lyrics()
print(f"Loaded {len(df)} songs.")

# ================================
# 1. TF-IDF Saving and Loading
# ================================
if os.path.exists(TFIDF_VECTORIZER_PATH) and os.path.exists(TFIDF_MATRIX_PATH):
    print("Loading saved TF-IDF vectorizer and matrix...")
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
else:
    print("Fitting and saving TF-IDF vectorizer and matrix...")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['lyrics'])
    joblib.dump(tfidf_vectorizer, TFIDF_VECTORIZER_PATH)
    joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)

def recommend_tfidf(query_title_artist, top_n=5):
    idx = df[df['title'] == query_title_artist].index[0]
    query_vec = tfidf_matrix[idx]
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[::-1][1:top_n+1]
    print(f"\nTF-IDF Recommendations for: {query_title_artist}")
    for i in similar_indices:
        print(f"- {df.iloc[i]['title']}")

# ================================
# 2. BERT Saving and Loading
# ================================
def load_or_download_bert_model():
    if os.path.exists(BERT_MODEL_PATH):
        print(f"Loading BERT model from {BERT_MODEL_PATH}")
        return SentenceTransformer(BERT_MODEL_PATH)
    else:
        print("Downloading BERT model from Hugging Face...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"Saving BERT model to {BERT_MODEL_PATH}")
        model.save(BERT_MODEL_PATH)
        return model

bert_model = load_or_download_bert_model()
valid_df = df[df['lyrics'].str.strip().astype(bool)].copy()
if os.path.exists(BERT_EMBEDDINGS_PATH):
    print("Loading BERT embeddings from file...")
    bert_embeddings = np.load(BERT_EMBEDDINGS_PATH)
    if len(bert_embeddings) != len(valid_df):
        print("Mismatch found! Regenerating BERT embeddings...")
        bert_embeddings = bert_model.encode(valid_df['lyrics'].tolist(), show_progress_bar=True)
        np.save(BERT_EMBEDDINGS_PATH, bert_embeddings)
else:
    print("Generating BERT embeddings...")
 
    bert_embeddings = bert_model.encode(valid_df['lyrics'].tolist(), show_progress_bar=True)
    if len(bert_embeddings) != len(valid_df):
        raise ValueError("Mismatch between embeddings and valid lyrics!")
    
    np.save(BERT_EMBEDDINGS_PATH, bert_embeddings)

# df['bert_vec'] = list(bert_embeddings)
valid_df['bert_vec'] = list(bert_embeddings)
df = df.merge(valid_df[['title', 'bert_vec']], on='title', how='left')

def recommend_bert(query_title_artist, top_n=5):
    query_idx = df[df['title'] == query_title_artist].index[0]
    query_vec = df.iloc[query_idx]['bert_vec']
    cosine_sim = cosine_similarity([query_vec], bert_embeddings).flatten()
    similar_indices = cosine_sim.argsort()[::-1][1:top_n+1]
    print(f"\nBERT Recommendations for: {query_title_artist}")
    for i in similar_indices:
        print(f"- {df.iloc[i]['title']}")

# ================================
# Test Recommendation
# ================================
if __name__ == "__main__":
    test_song = df['title'].iloc[0]  # Example
    print("**************************************************")
    print(df.columns)
    recommend_tfidf(test_song)
    recommend_bert(test_song)
