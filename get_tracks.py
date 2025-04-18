
import os
import json
import time
import logging
import requests
from urllib.parse import urlencode
# from flask import Flask, request, redirect, jsonify, render_template_string
from flask import Flask, request, jsonify ,render_template_string, render_template , redirect, Response
import uuid
import logging
import requests
from flask import jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_user_top_tracks(token):
    url = 'https://api.spotify.com/v1/me/top/tracks'
    headers = {
        'Authorization': f'Bearer {token}'
    }
    params = {
        'time_range': 'medium_term',  # Can be 'short_term', 'medium_term', or 'long_term'
        'limit': 10,                  # Number of items to return
        'offset': 10                  # Offset to use for pagination
    }
    
    # Make the API call to Spotify
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        # Parse the response JSON if successful
        data = response.json()
        logger.info(f"Top artists data for user: {data}")
        track_data = data.get("items", {})
        
        track = track_data[0]

# Track level
        track_name = track['name']
        track_id = track['id']
        track_uri = track['uri']
        track_popularity = track['popularity']
        track_duration_ms = track['duration_ms']
        track_number = track['track_number']
        track_explicit = track['explicit']
        track_href = track['href']
        track_spotify_url = track['external_urls']['spotify']
        track_preview_url = track['preview_url']
        print('#############################')
        print(track_uri)
        # ISRC (external ID)
        track_isrc = track['external_ids']['isrc']

        # Album level
        album = track['album']
        album_name = album['name']
        album_id = album['id']
        album_uri = album['uri']
        album_release_date = album['release_date']
        album_total_tracks = album['total_tracks']
        album_spotify_url = album['external_urls']['spotify']
        album_images = album['images']  # List of image dicts

        # Artists on track
        track_artists = track['artists']
        artist_names = [artist['name'] for artist in track_artists]
        artist_ids = [artist['id'] for artist in track_artists]
        artist_uris = [artist['uri'] for artist in track_artists]

        # First artist details (optional)
        first_artist = track_artists[0]
        first_artist_name = first_artist['name']
        first_artist_id = first_artist['id']
        first_artist_uri = first_artist['uri']
        first_artist_spotify_url = first_artist['external_urls']['spotify']

        # Album artist(s)
        album_artists = album['artists']
        album_artist_names = [a['name'] for a in album_artists]

        # Album images - largest image URL
        album_image_url = album_images[0]['url'] if album_images else None
        
        # logger.info(f"$$$$$$$$$$Top artists track for user: {track}")
        return data
    else:
        # Log the error if the API call fails
        logger.error(f"Failed to fetch top artists: {response.status_code} - {response.text}")
        return None

def get_followed_artists(access_token, limit=20):
    url = "https://api.spotify.com/v1/me/following"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "type": "artist",
        "limit": limit
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        artists = data.get("artists", {}).get("items", [])
        print("Followed Artists:")
        for artist in artists:
            print(f"- {artist['name']} (Followers: {artist['followers']['total']})")
        return artists
    else:
        print(f"Failed to fetch followed artists: {response.status_code}")
        print(response.text)
        return None
    