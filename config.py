import os

# Directory paths
LYRICS_DIR = 'lyrics'
SAVE_DIR = 'saved_models'
os.makedirs(SAVE_DIR, exist_ok=True)

# File paths for saved models and data
TFIDF_VECTORIZER_PATH = os.path.join(SAVE_DIR, 'tfidf_vectorizer.pkl')
TFIDF_MATRIX_PATH = os.path.join(SAVE_DIR, 'tfidf_matrix.pkl')
BERT_MODEL_PATH = os.path.join(SAVE_DIR, 'bert_model')
BERT_EMBEDDINGS_PATH = os.path.join(SAVE_DIR, 'bert_embeddings.npy')
