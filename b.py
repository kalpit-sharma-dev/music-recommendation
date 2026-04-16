import os
import json
import time
import logging
import requests
import pickle
import pandas as pd
from urllib.parse import urlencode
from flask import Flask, request, jsonify, render_template_string, render_template, redirect, Response
import uuid
import joblib
import numpy as np
from flask_cors import CORS
from get_tracks import get_followed_artists, get_user_top_tracks
from datetime import datetime
import re
from bs4 import BeautifulSoup
import threading
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import random
from config import *

# Big Data Imports
from pymongo import MongoClient
from pyspark.sql import SparkSession
from kafka import KafkaConsumer, KafkaProducer
from redis import Redis

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Big Data Components
def init_big_data():
    # MongoDB Setup
    mongo_client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017/'))
    db = mongo_client['music_recommender']
    
    # Spark Setup
    spark = SparkSession.builder \
        .appName("MusicRecommender") \
        .config("spark.mongodb.input.uri", os.getenv('MONGO_URI', 'mongodb://localhost:27017/music_recommender.tracks')) \
        .config("spark.mongodb.output.uri", os.getenv('MONGO_URI', 'mongodb://localhost:27017/music_recommender.recommendations')) \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    # Kafka Setup
    kafka_producer = KafkaProducer(
        bootstrap_servers=os.getenv('KAFKA_BROKERS', 'localhost:9092').split(','),
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    # Redis Setup
    redis_cache = Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=int(os.getenv('REDIS_DB', 0)))
    
    return db, spark, kafka_producer, redis_cache

# Initialize
db, spark, kafka_producer, redis_cache = init_big_data()

app = Flask(__name__)
CORS(app)

# NLP Setup
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))
LYRICS_DIR = 'lyrics'
os.makedirs(LYRICS_DIR, exist_ok=True)

# Load models
with open('music_recommender.pkl', 'rb') as f:
    data = pickle.load(f)
    cosine_sim = data['cosine_sim']
    df = data['df']

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Kafka Consumer for real-time events
def start_kafka_consumer():
    consumer = KafkaConsumer(
        'user-listening-events',
        bootstrap_servers=os.getenv('KAFKA_BROKERS', 'localhost:9092').split(','),
        auto_offset_reset='earliest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8')))
    
    while True:
        msg = consumer.poll(1.0)
    
        if msg is None:
         continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
        else:
            logger.error(f"Kafka error: {msg.error()}")
            break  
    try:
        event = json.loads(msg.value().decode('utf-8'))
        # Process your event here
    except Exception as e:
        logger.error(f"Error processing message: {e}")

# Start Kafka consumer in background
threading.Thread(target=start_kafka_consumer, daemon=True).start()

# Spark Recommendation Engine
class SparkRecommender:
    def __init__(self, spark):
        self.spark = spark
        self.model = None
        
    def train(self):
        # Load interaction data from MongoDB
        df = self.spark.read.format("mongo").option("uri", 
            os.getenv('MONGO_URI', 'mongodb://localhost:27017/music_recommender.user_interactions')).load()
        
        # Train ALS model
        from pyspark.ml.recommendation import ALS
        als = ALS(
            maxIter=5,
            regParam=0.01,
            userCol="user_id",
            itemCol="track_id",
            ratingCol="interaction_score",
            coldStartStrategy="drop")
        
        self.model = als.fit(df)
        
    def recommend_for_user(self, user_id, num_recs=10):
        if not self.model:
            self.train()
            
        # Create dataframe with user ID
        user_df = self.spark.createDataFrame([(user_id,)], ["user_id"])
        
        # Generate recommendations
        recs = self.model.recommendForUserSubset(user_df, num_recs)
        
        # Convert to Python dict
        return [row.track_id for row in recs.collect()[0].recommendations]

spark_recommender = SparkRecommender(spark)

# Hybrid Recommendation System
def hybrid_recommendations(user_id, track_name=None, num_recs=10):
    # Check cache first
    cache_key = f"recs:{user_id}:{track_name if track_name else 'top'}"
    cached = redis_cache.get(cache_key)
    if cached:
        return pickle.loads(cached)
    
    # Collaborative filtering from Spark
    cf_recs = spark_recommender.recommend_for_user(user_id, num_recs)
    
    # Content-based recommendations
    if track_name:
        cb_recs = get_recommendation(track_name).to_dict('records')
    else:
        # Get user's top tracks from Spotify
        user_data = db.users.find_one({"user_id": user_id})
        if user_data and 'access_token' in user_data:
            top_tracks = get_user_top_tracks(user_data['access_token'])
            cb_recs = [{'track_id': t['id'], 'name': t['name']} for t in top_tracks.get('items', [])]
        else:
            cb_recs = []
    
    # Combine results (simple merge for demo)
    combined = {
        'collaborative': cf_recs[:num_recs//2],
        'content_based': cb_recs[:num_recs//2]
    }
    
    # Cache results
    redis_cache.setex(cache_key, 300, pickle.dumps(combined))  # 5 minute cache
    
    return combined

# Updated Routes with Big Data Integration
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    track_name = request.args.get('track')
    
    if not user_id:
        return jsonify({'error': 'User ID required'}), 400
    
    recommendations = hybrid_recommendations(user_id, track_name)
    
    # Enrich with Spotify data
    user_data = db.users.find_one({"user_id": user_id})
    if not user_data or 'access_token' not in user_data:
        return jsonify({'error': 'User not authenticated'}), 401
    
    enriched_recs = []
    for rec_type, recs in recommendations.items():
        for rec in recs:
            track_id = rec.get('track_id') if isinstance(rec, dict) else rec
            track_info = get_track_details([track_id], user_data['access_token'])
            if track_info:
                enriched_recs.append({
                    'type': rec_type,
                    'track': track_info[0] if isinstance(track_info, list) else track_info
                })
    
    return jsonify({'recommendations': enriched_recs})

@app.route('/track_played', methods=['POST'])
def track_played():
    data = request.get_json()
    user_id = data.get('user_id')
    track_id = data.get('track_id')
    
    if not user_id or not track_id:
        return jsonify({'error': 'Missing user_id or track_id'}), 400
    
    # Send event to Kafka
    event = {
        'user_id': user_id,
        'track_id': track_id,
        'timestamp': datetime.utcnow().isoformat(),
        'event_type': 'track_played'
    }
    
    try:
        kafka_producer.send('user-listening-events', value=event)
        kafka_producer.flush()
        return jsonify({'status': 'event queued'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Existing functions with MongoDB integration
@app.route('/callback')
def callback():
    # ... existing code ...
    
    # Save user data to MongoDB
    db.users.update_one(
        {'user_id': user_id},
        {'$set': {
            'access_token': access_token,
            'refresh_token': data.get('refresh_token'),
            'profile': profile,
            'last_updated': datetime.utcnow()
        }},
        upsert=True
    )
    
    return redirect(f'/show_token?access_token={access_token}&user_id={user_id}')

def update_realtime_recommendations(user_id, track_id):
    """Update recommendations based on real-time event"""
    # Increment interaction score in MongoDB
    db.user_interactions.update_one(
        {'user_id': user_id, 'track_id': track_id},
        {'$inc': {'interaction_score': 1}},
        upsert=True
    )
    
    # Invalidate cache for this user
    redis_cache.delete(f"recs:{user_id}:*")

# Existing utility functions (get_recommendation, search_track, etc.) remain the same
# ...

if __name__ == '__main__':
    logger.info("Starting Flask server with Big Data integration...")
    app.run(host='0.0.0.0', port=5000, debug=True,ssl_context='adhoc')