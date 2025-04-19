import os
import re
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

### PATH TO YOUR LYRICS FOLDER ###
LYRICS_DIR = 'lyrics'

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
                title_artist = filename[:-4]  # remove .txt
                data.append({
                    'title_artist': title_artist,
                    'lyrics': cleaned_lyrics
                })
    return pd.DataFrame(data)

df = load_lyrics()
print(f"Loaded {len(df)} songs.")

# ===============================
# 1. TF-IDF Recommendation
# ===============================
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['lyrics'])

def recommend_tfidf(query_title_artist, top_n=5):
    idx = df[df['title_artist'] == query_title_artist].index[0]
    query_vec = tfidf_matrix[idx]
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[::-1][1:top_n+1]
    print(f"\nTF-IDF Recommendations for: {query_title_artist}")
    for i in similar_indices:
        print(f"- {df.iloc[i]['title_artist']}")

# ===============================
# 2. BERT Embedding (DistilBERT)
# ===============================
print("Generating BERT embeddings...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
df['bert_vec'] = list(bert_model.encode(df['lyrics'], show_progress_bar=True))

def recommend_bert(query_title_artist, top_n=5):
    query_vec = df[df['title_artist'] == query_title_artist]['bert_vec'].values[0]
    all_vecs = list(df['bert_vec'])
    cosine_sim = cosine_similarity([query_vec], all_vecs).flatten()
    similar_indices = cosine_sim.argsort()[::-1][1:top_n+1]
    print(f"\nBERT Recommendations for: {query_title_artist}")
    for i in similar_indices:
        print(f"- {df.iloc[i]['title_artist']}")

# ===============================
# 🔍 Try it out
# ===============================
if __name__ == "__main__":
    test_song = df['title_artist'].iloc[0]  # pick the first song for demo

    recommend_tfidf(test_song)
    recommend_bert(test_song)
