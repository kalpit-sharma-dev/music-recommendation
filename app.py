import os
import json
import time
import logging
import requests
import pickle
import pandas as pd
from urllib.parse import urlencode
# from flask import Flask, request, redirect, jsonify, render_template_string
from flask import Flask, request, jsonify ,render_template_string, render_template , redirect, Response
import uuid
import logging
import requests
from flask import jsonify, request ,send_from_directory
from flask import Flask
from flask_cors import CORS
from get_tracks import get_followed_artists
import csv
from datetime import datetime
import re
  # Enable CORS for all routes



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from get_tracks import get_user_top_tracks

app = Flask(__name__)
CORS(app)
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Spotify API 

# CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
# CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
CLIENT_ID = "1b51b0e34b4a4da2a78dd5bd9d1d7e02"
CLIENT_SECRET = "3450c693ec4541e980faa7bac14844c7"
REDIRECT_URI = 'http://127.0.0.1:5000/callback'
SCOPE = 'user-top-read user-modify-playback-state streaming user-read-email user-read-private user-follow-read'

with open('music_recommender.pkl', 'rb') as f:
    data = pickle.load(f)
    cosine_sim = data['cosine_sim']
    df = data['df']

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def get_recommendation(title, cosine_sim=cosine_sim, df=df):
    try:
        idx = df[df['song_name'] == title].index[0]
    except IndexError:
        return pd.DataFrame()  # Return empty if song not found
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    song_indices = [i[0] for i in sim_scores]
    
    return df[['song_name', 'singer', 'duration', 'popularity', 'Stream']].iloc[song_indices]


# In-memory store for user data
user_tokens = {}
user_listening_data = {}

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/signupindex')
def signupindex():
    return render_template("signup.html")

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    name = data.get('name')

    if not email or not name:
        return jsonify({'error': 'Missing name or email'}), 400

    # Here you would store the user in a real database. We're simulating it.
    user_id = email  # Using email as user_id for simplicity
    logger.info(f"Registered new user: {name} ({email})")

    # Redirect user to Spotify login with user_id
    params = urlencode({
        'client_id': CLIENT_ID,
        'response_type': 'code',
        'redirect_uri': REDIRECT_URI,
        'scope': SCOPE,
        'state': user_id,  # Pass our app's user_id via state
    })
    return jsonify({
        'auth_url': f'https://accounts.spotify.com/authorize?{params}',
        'user_id': user_id
    })

@app.route('/login')
def login():
    # Generate a unique user_id for the user (can be persisted in your DB or session)
    # user_id = "m24de3042@iitj.ac.in"
    user_id = str(uuid.uuid4())
    logger.info(f"Generated user_id: {user_id}")

    # Save the user_id temporarily (you can also store it in a session or database)
    user_tokens[user_id] = None  # Initialize user entry
    
    # Prepare the authorization URL with the user_id in the state
    params = urlencode({
        'client_id': CLIENT_ID,
        'response_type': 'code',
        'redirect_uri': REDIRECT_URI,
        'scope': SCOPE,
        'state': user_id  # Pass the user_id as state
    })
    
    logger.info(f"Redirecting to Spotify login with state: {user_id}")
    return redirect(f'https://accounts.spotify.com/authorize?{params}')

@app.route('/callback')
def callback():
    code = request.args.get('code')
    state = request.args.get('state')  # Retrieve the state (user_id) from the query parameters
    if not code or not state:
        logger.warning("Callback received with no code or state")
        return 'No code or state provided', 400

    logger.info(f"Received code and state: {state}, requesting access token")
    
    # Retrieve the stored user_id and ensure it matches the state
    if state not in user_tokens:
        logger.warning(f"Invalid state (user_id) received: {state}")
        return 'Invalid state', 400

    # Request the access token from Spotify
    response = requests.post('https://accounts.spotify.com/api/token', data={
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': REDIRECT_URI,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    })

    data = response.json()
    access_token = data.get('access_token')
    if not access_token:
        logger.error(f"Failed to get access token: {data}")
        return 'Failed to get access token', 400

    headers = {'Authorization': f'Bearer {access_token}'}
    profile_response = requests.get('https://api.spotify.com/v1/me', headers=headers)
    print(profile_response.text)
    if profile_response.status_code != 200:
        logger.error(f"Failed to fetch profile: {profile_response.status_code}, {profile_response.text}")
        return 'Failed to fetch user profile', 400

    profile = profile_response.json()
    user_id = profile['id']
    logger.info(f"Logged in as user: {user_id}")

    # Save the access token for the user
    user_tokens[state] = access_token  # state here is used as user_id
    # return redirect(f'/index2.html?access_token={access_token}&user_id={user_id}')
    return redirect(f'/show_token?access_token={access_token}&user_id={user_id}')
    # return redirect(f'http://127.0.0.1:5000/index2.html?access_token={access_token}&user_id={state}')

@app.route('/show_token')
def show_token():
    access_token = request.args.get('access_token')
    user_id = request.args.get('user_id')
    return render_template('index2.html', access_token=access_token, user_id=user_id)

@app.route('/collect_user_data')
def collect_user_data():
    user_id = request.args.get('user_id')
    logger.info(f"Collecting user data for user_id: {user_id}")
    token = user_tokens.get(user_id)
    if not token:
        logger.warning(f"Unauthorized access attempt for user_id: {user_id}")
        return 'Unauthorized', 401

    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get('https://api.spotify.com/v1/me/top/tracks?limit=50', headers=headers)
    top_tracks = response.json().get('items', [])

    track_features = []
    for track in top_tracks:
        track_features.append({
            'id': track['id'],
            'name': track['name'],
            'preview_url': track.get('preview_url'),
            'genres': [],
            'type': track['album']['album_type'],
        })

    user_listening_data[user_id] = track_features
    logger.info(f"Collected {len(track_features)} tracks for user_id: {user_id}")
    return 'Data collected'

@app.route('/recommendations', methods=['POST'])
def getRecommendations():
    data = request.get_json()
    user_id = data.get('user_id')
    token = data.get('access_token')
    logger.info(f"Received request for recommendations with user_id: {user_id}")
    
    # Fetch the listening data for the user, return empty if not found
    # Example usage
    # user_token = 'your_spotify_access_token_here'
    
    # top_artists = get_user_top_tracks(token)
    top_artists = get_followed_artists(token)
    if top_artists:
        logger.info(f"Top artists: {top_artists}")
    else:
        logger.warning("Failed to retrieve top artists.")
    # data = user_listening_data.get(user_id)
    data=get_user_top_tracks(token=token)
    track_data = data.get("items", {})
    filtered_tracks = [
        {
            'name': track['name'],
            'url': track['external_urls']['spotify'],
            'id': track['id']
        }
        for track in track_data
    ]
    # if not data:
    #     logger.warning(f"No listening data found for user_id: {user_id}")
    #     return jsonify({'recommended_tracks': []})
    
    # # Get the token for authorization
    # token = user_tokens.get(user_id)
    # if not token:
    #     logger.warning(f"No token found for user_id: {user_id}")
    #     return jsonify({'recommended_tracks': []})
    
    # headers = {'Authorization': f'Bearer {token}'}
    # logger.info(f"Token retrieved for user_id: {user_id}")

    # # Collect track IDs (only those that have an 'id' field)
    # track_ids = [track['id'] for track in data if track.get('id')]
    # if not track_ids:
    #     logger.warning(f"No valid track IDs found for user_id: {user_id}")
    #     return jsonify({'recommended_tracks': []})

    # logger.info(f"Found {len(track_ids)} track IDs for user_id: {user_id}")

    # # Fetch audio features in chunks of 100 (Spotify's limit)
    # all_features = []
    # for i in range(0, len(track_ids), 100):
    #     chunk = track_ids[i:i+100]
        
    #     # Make the request to fetch the audio features
    #     logger.info(f"Fetching audio features for chunk: {chunk}")
    #     features_resp = requests.get(
    #         'https://api.spotify.com/v1/audio-features',
    #         headers=headers,
    #         params={'ids': ','.join(chunk)}
    #     )
        
    #     # Check if the API response is successful
    #     if features_resp.status_code != 200:
    #         logger.error(f"Failed to fetch audio features for chunk: {chunk}, Status code: {features_resp.status_code}")
    #         return jsonify({'recommended_tracks': []})
        
    #     features_data = features_resp.json().get('audio_features', [])
    #     logger.info(f"Fetched {len(features_data)} audio features for chunk: {chunk}")
    #     all_features.extend([f for f in features_data if f])

    # # Helper function to calculate the average of a given key from the features
    # def avg(key):
    #     values = [f[key] for f in all_features if f and key in f and f[key] is not None]
    #     return sum(values) / len(values) if values else None

    # # Calculate the averages for energy, valence, and danceability
    # target_energy = avg('energy')
    # target_valence = avg('valence')
    # target_danceability = avg('danceability')

    # # Log the calculated averages
    # logger.info(f"Calculated averages -> Energy: {target_energy}, Valence: {target_valence}, Danceability: {target_danceability}")

    # # Fallback if any of the averages are missing
    # if None in (target_energy, target_valence, target_danceability):
    #     logger.warning(f"Missing average values for energy, valence, or danceability. Returning empty recommendations.")
    #     return jsonify({'recommended_tracks': []})

    # # Request recommendations based on the user's preferences
    # logger.info("Requesting Spotify recommendations based on the user's listening data and audio features.")
    # rec_res = requests.get('https://api.spotify.com/v1/recommendations', headers=headers, params={
    #     'limit': 10,
    #     'seed_tracks': ','.join(track_ids[:5]),  # Using top 5 tracks as seeds
    #     'target_energy': target_energy,
    #     'target_valence': target_valence,
    #     'target_danceability': target_danceability,
    # })

    # # Check if the recommendation API response is successful
    # if rec_res.status_code != 200:
    #     logger.error(f"Failed to fetch recommendations, Status code: {rec_res.status_code}")
    #     return jsonify({'recommended_tracks': []})

    # tracks = rec_res.json().get('tracks', [])
    # logger.info(f"Fetched {len(tracks)} recommended tracks from Spotify.")

    # # Filter out tracks that do not have a preview URL
    # track_data = [{
    #     'id': track['id'],
    #     'name': track['name'],
    #     'preview_url': track.get('preview_url')
    # } for track in tracks if track.get('preview_url')]

    logger.info(f"Returning {len(filtered_tracks)} recommended tracks with preview URLs.")
    
    logger.info(f"Returning Full Track List {filtered_tracks} recommended tracks with preview URLs.")

    return jsonify({'recommended_tracks': filtered_tracks})

@app.route('/recommend', methods=['GET'])
def recommend():
    song_name = request.args.get('song')
    if not song_name:
        return jsonify({'error': 'Please provide a song name'}), 400

    recommendations = get_recommendation(song_name)
    print("Columns in recommendations:", recommendations.columns)
    if recommendations.empty:
        return jsonify({'error': 'Song not found in database'}), 404

    # Add Spotify data to each recommended song
    access_token = "BQDKUxaFBKXsrVuC27lCIe2ioo26Gko0rpQq6hWmhEnx7jMNHxVkcvgagwp6wINRmtwAVlm6T5xr7TX6WfBodi4vQul2FpZHt0Tqv3PchSaHSqrdC06c2FwUmIvqMaocPwT3OL-JeeveKmgOFCJqcEaU_kuLCDbQEtRsONpwNIWOEtIpdLRpG6n_yqlt99L2MD4GDHxVQS2832VnkxFgKilMOR8oBwMKiL4zKnoIn74Oc85WwK55MhN6FR8VT3GTGo25Ei8O"  # Use valid token from OAuth
    enriched_results = []

    for _, row in recommendations.iterrows():
        track_info = search_track(row['song_name'], access_token)
        enriched_results.append({
            'track_name': row['song_name'],
            'artist': row.get('artist', ''),
            'url': track_info.get('url', None),
            'id': track_info.get('id', None)
        })

    return jsonify(enriched_results)


def remove_brackets(text):
    """
    Remove brackets and anything inside them from a string
    
    Args:
        text (str): Input string containing brackets
        
    Returns:
        str: Cleaned string without brackets or their content
    """
    return re.sub(r'\s*\([^)]*\)', '', text).strip()


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    access_token = data.get('token')
    track_name = data.get('query')
    print(access_token)
    print(track_name)
    recommendations = get_recommendation(track_name)
    
    tracks = []
    for _, row in recommendations.iterrows():    
        url = "https://api.spotify.com/v1/search"
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        params = {
            # "q": track_name,
            "q": remove_brackets(row['song_name']),
            "type": "track",
            "limit": 1
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            results = response.json()
            for item in results['tracks']['items']:
                tracks.append({
                'id': item['id'],
                'name': item['name'],
                'artist': ', '.join([a['name'] for a in item['artists']]),
                'url': item['external_urls']['spotify']
            })
        else:
            print('Spotify API error:', response.status_code, response.text)
    print('Spotify :', tracks)
    # track_ids = [track['id'] for track in tracks]
    # get_track_details(track_ids,access_token)
    return jsonify({'tracks': tracks})

def search_track(track_name, access_token):
    url = "https://api.spotify.com/v1/search"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "q": track_name,
        "type": "track",
        "limit": 5
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        return {}

    items = response.json().get('tracks', {}).get('items', [])
    if not items:
        return {}

    track = items[0]
    return {
        'url': track['external_urls']['spotify'],
        'id': track['id'],
        'name':track['name'],
        'artist': ', '.join([a['name'] for a in track['artists']]),
    }

@app.route('/player/<track_id>')
def player(track_id):
    logger.info(f"Serving player for track_id: {track_id}")
    return render_template_string("""
    <html>
    <head><title>Music Player</title></head>
    <body>
      <h2>Playing Track: {{track_id}}</h2>
      <iframe src="https://open.spotify.com/embed/track/{{track_id}}" width="300" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
    </body>
    </html>
    """, track_id=track_id)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


def format_duration(ms):
    """Convert milliseconds to MM:SS format"""
    seconds = int(ms / 1000)
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

# @app.route('/api/track-details', methods=['GET'])
def get_track_details(tracks,access_token):
    # Get access token from Authorization header
    # auth_header = request.headers.get('Authorization')
    # if not auth_header:
    #     return jsonify({'error': 'Authorization header missing'}), 401
    
    # Extract token
    # token = auth_header.split(' ')[1] if ' ' in auth_header else auth_header
    
    if isinstance(tracks, list):
            track_ids = ','.join(tracks)
    
    # Get track IDs from query parameters
    # track_ids = request.args.get('ids')
    # if not track_ids:
    #     return jsonify({'error': 'Track IDs parameter missing'}), 400
    
    try:
        # First API call - Get track details
        tracks_url = f'https://api.spotify.com/v1/tracks?ids={track_ids}'
        tracks_response = requests.get(
            tracks_url,
            headers={'Authorization': f'Bearer {access_token}'}
        )
        
        if tracks_response.status_code != 200:
            return jsonify({
                'error': 'Failed to fetch track details',
                'status_code': tracks_response.status_code,
                'message': tracks_response.json().get('error', {}).get('message')
            }), tracks_response.status_code
        
        # Second API call - Get audio features
        features_url = f'https://api.spotify.com/v1/audio-features?ids={track_ids}'
        features_response = requests.get(
            features_url,
            headers={'Authorization': f'Bearer {access_token}'}
        )
        print(features_response.json().get('error', {}).get('message'))
        if features_response.status_code != 200:
            return jsonify({
                'error': 'Failed to fetch audio features',
                'status_code': features_response.status_code,
                'message': features_response.json().get('error', {}).get('message')
            }), features_response.status_code
        
        # Process responses
        tracks_data = tracks_response.json().get('tracks', [])
        features_data = {f['id']: f for f in features_response.json().get('audio_features', []) if f}
        
        # Prepare CSV data
        csv_data = []
        headers = [
            'song_name', 'singer', 'singer_id', 'duration', 'language', 
            'released_date', 'danceability', 'acousticness', 'energy', 
            'liveness', 'loudness', 'speechiness', 'tempo', 'mode', 
            'key', 'Valence', 'time_signature', 'popularity', 'Stream'
        ]
        
        for track in tracks_data:
            if not track:
                continue
                
            track_id = track.get('id')
            features = features_data.get(track_id, {})
            
            # Format release date (assuming Spotify returns YYYY-MM-DD)
            release_date = track.get('album', {}).get('release_date', '')
            if release_date:
                try:
                    dt = datetime.strptime(release_date, '%Y-%m-%d')
                    release_date = dt.strftime('%d-%m-%Y')
                except ValueError:
                    pass
            
            csv_row = {
                'song_name': track.get('name', ''),
                'singer': '|'.join([artist.get('name', '') for artist in track.get('artists', [])]),
                'singer_id': '|'.join([f"/artist/{artist.get('id', '')}" for artist in track.get('artists', [])]),
                'duration': format_duration(features.get('duration_ms', 0)),
                'language': 'Hindi',
                'released_date': release_date,
                'danceability': features.get('danceability', 0),
                'acousticness': features.get('acousticness', 0),
                'energy': features.get('energy', 0),
                'liveness': features.get('liveness', 0),
                'loudness': features.get('loudness', 0),
                'speechiness': features.get('speechiness', 0),
                'tempo': features.get('tempo', 0),
                'mode': features.get('mode', 0),
                'key': features.get('key', 0),
                'Valence': features.get('valence', 0),
                'time_signature': features.get('time_signature', 4),
                'popularity': track.get('popularity', 0),
                'Stream': 0  # Placeholder since Spotify doesn't provide stream count
            }
            csv_data.append(csv_row)
        
        # Generate CSV filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'spotify_hindi_tracks_{timestamp}.csv'
        
        # Write to CSV file
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(csv_data)
        
        return jsonify({
            'message': f'CSV file generated successfully: {filename}',
            'data': csv_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True)