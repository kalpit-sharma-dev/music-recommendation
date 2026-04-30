# Music Recommendation Project Viva Guide

This document is a quick viva-ready reference for your project codebase (`app.py`, `appp.py`) and architecture choices.

## Which File Is Actually Used?

- **Yes, `app.py` is the main file used to run your project backend**.
- It contains the complete Flask server, Spotify OAuth routes, recommendation endpoints, lyrics pipeline, and big-data integration.
- Typical run command is:
  - `python app.py`
- `appp.py` is a smaller/older simplified version with limited routes (mainly a basic `/recommend` flow).

## Project in One Minute

- This is a Flask-based music recommendation system integrated with Spotify APIs.
- It combines:
  - **Content-based recommendation** using song metadata similarity (`cosine_sim` from precomputed model)
  - **Collaborative filtering** using Spark ALS on user interaction data
  - **Lyrics-based recommendation** using TF-IDF and BERT embeddings
- It also includes a basic big-data pipeline:
  - MongoDB for storing user interactions
  - Kafka for streaming user events
  - Redis for caching recommendation responses

## Why These Choices (Important Viva Points)

### Why both content-based and collaborative?
- **Content-based** handles cold-start for items, and works well when item metadata/features are available.
- **Collaborative filtering** captures behavior patterns from many users (what similar users listen to).
- Using both in a **hybrid approach** improves recommendation quality and robustness.

### Why cache?
- Recommendation pipelines and API calls are expensive and repeated often.
- Redis caching reduces response time and backend load.
- It improves user experience by returning frequent requests quickly.

### Why Kafka + Mongo + Spark?
- **Kafka**: real-time ingestion of user listening/interactions as events.
- **MongoDB**: persistent storage for interactions and user-level data.
- **Spark**: scalable distributed training/inference (ALS collaborative filtering).
- Together, they separate ingest, storage, and compute cleanly.

### Why TF-IDF and BERT both?
- **TF-IDF**: fast, lightweight, lexical similarity baseline.
- **BERT embeddings**: semantic similarity, better contextual understanding of lyrics.
- Keeping both allows speed-vs-quality tradeoff and comparison.

## Architecture Flow (Explain Like a Pipeline)

1. User logs in via Spotify OAuth (`/login`, `/callback`).
2. User searches or requests recommendations (`/search_music`, `/recommend`, `/getlyricbert`, `/getlyrictfidf`).
3. System fetches Spotify data and optionally logs interactions.
4. Interaction events are sent to Kafka and stored/upserted in MongoDB.
5. Spark ALS can train on interactions and generate collaborative recommendations.
6. Content/lyrics recommendations are generated via cosine/TF-IDF/BERT.
7. Hybrid results are cached in Redis for low latency.

## Notebook (`.ipynb`) Files Included

Your project also has notebook-based work used for experimentation/training:

- `hindi_recommendation.ipynb`
- `music-recommendation-system-using-spotify-dataset.ipynb`

How to explain in viva:
- Notebooks were used for **exploration, preprocessing, and model experimentation/prototyping**.
- Production/API serving logic was moved into `app.py` for deployment and endpoint integration.
- Final reusable artifacts (for example model/vectorizer files) are loaded by the Flask app.

## `app.py` Line-by-Line Explanation (In Order)

Use this section when your professor asks: "Explain code line by line."  
It is written in exact file order, grouped into short line ranges for quick viva delivery.

### Lines 1-43: Imports and Dependencies
- Import Python core modules (`os`, `json`, `time`, `uuid`, `threading`, `random`, etc.).
- Import Flask request/response utilities and template rendering helpers.
- Import ML/NLP libraries (`pandas`, `numpy`, `pickle`, `joblib`, `nltk`, `TfidfVectorizer`, `SentenceTransformer`).
- Import Spotify/helper functions from local modules (`get_tracks`, `config`).
- Import big-data stack (`MongoClient`, `SparkSession`, `Kafka Producer/Consumer`, `Redis`).

### Lines 44-49: Flask and Logging Initialization
- Create Flask app instance.
- Enable CORS for cross-origin frontend requests.
- Configure global logging level and logger object.

### Lines 51-60: `_NoopRedis` Fallback Class
- Defines safe dummy cache methods: `get`, `setex`, `delete`.
- Used when Redis is unavailable so code does not crash.

### Lines 63-108: `init_big_data()`
- Reads Mongo URI from environment (or default URI).
- Connects to MongoDB and validates connection with `server_info()`.
- Builds Spark session and configures Mongo input/output collections.
- Creates Kafka producer with broker config.
- Creates Redis client.
- Returns all initialized services.
- If anything fails, logs warning and returns local-mode fallbacks.

### Lines 110-115: Global Service Setup
- Calls `init_big_data()` and stores globals (`db`, `spark`, `kafka_producer`, `redis_cache`).
- Sets `BIG_DATA_ENABLED` boolean based on successful initialization.
- Recreates Flask app + CORS (duplicate initialization in file).

### Lines 117-160: Request/Response Logging Middleware
- `before_request`: assigns request start time + short request ID.
- Extracts query/body payload and redacts sensitive token fields before logging.
- Logs method/path/query/payload for traceability.
- `after_request`: computes duration and logs status code with request ID.

### Lines 163-177: NLP Setup + Base Model Loading
- Downloads NLTK stopwords.
- Creates stopword set and ensures `lyrics` folder exists.
- Loads `music_recommender.pkl` (`cosine_sim`, `df`).
- Loads `tfidf_vectorizer.pkl`.

### Lines 179-207: Kafka Consumer Worker
- Creates Kafka consumer and subscribes to `user-listening-events`.
- Polls continuously, handles EOF and error cases.
- Decodes JSON event payload.
- Starts this consumer in a daemon thread only when big-data mode is enabled.

### Lines 210-245: Spark Collaborative Recommender Class
- `SparkRecommender.__init__`: stores Spark session and model placeholder.
- `train()`: loads interactions from Mongo and trains Spark ALS model.
- `recommend_for_user()`: lazy-trains model if missing and returns top-N recommendations.
- Creates `spark_recommender` instance.

### Lines 248-279: `hybrid_recommendations()`
- Builds user-specific cache key and checks Redis first.
- Gets collaborative recommendations from Spark (if enabled).
- Gets content-based recommendations from `get_recommendation()` or Spotify top tracks.
- Combines both recommendation types into one dictionary.
- Caches combined result for 5 minutes and returns it.

### Lines 281-293: Real-Time Update Helper
- `update_realtime_recommendations()` increments user-track interaction in Mongo.
- Invalidates cached recommendation entries for that user.

### Lines 294-304: Repeated NLP/Path Constants
- Re-downloads stopwords and recreates stopword set (duplicate block).
- Defines save directory and saved model file paths.

### Lines 307-320: Spotify Credentials + Duplicate Model Load
- Sets Spotify client credentials and redirect URI/scope.
- Loads `music_recommender.pkl` and `tfidf_vectorizer.pkl` again (duplicate).

### Lines 322-333: `get_recommendation()`
- Finds row index where `song_name == title`.
- If missing, returns empty DataFrame.
- Sorts cosine similarities and selects top 10 similar songs.
- Returns selected columns from dataframe.

### Lines 336-338: In-Memory Stores
- `user_tokens`: stores temporary user token mapping.
- `user_listening_data`: stores fetched listening history.

### Lines 340-347: Basic Page Routes
- `/` renders `index2.html`.
- `/signupindex` renders signup page.

### Lines 348-373: `/signup` Route
- Reads JSON body (`email`, `name`) and validates.
- Uses email as simple user_id.
- Builds Spotify authorization URL with `state=user_id`.
- Returns JSON containing auth URL and user_id.

### Lines 374-395: `/login` Route
- Generates UUID as temporary user_id.
- Stores it in `user_tokens`.
- Redirects user to Spotify authorization page with state.

### Lines 396-441: `/callback` Route
- Reads `code` and `state` from Spotify callback query.
- Validates presence and state integrity.
- Exchanges authorization code for access token using Spotify token API.
- Fetches profile (`/v1/me`) to verify token works.
- Saves token mapped to original `state`.
- Redirects to `/show_token` page with token + user id.

### Lines 443-447: `/show_token`
- Reads query params and renders `index2.html` with them.

### Lines 449-490: `/collect_user_data`
- Reads `user_id` and token.
- Fetches top tracks from Spotify.
- Builds list of track features (`id`, `name`, `preview_url`, etc.).
- Creates listening event payload and stores user track data in memory.
- Produces Kafka message if producer exists, then flushes.
- Returns success text.

### Lines 492-527: `/getMyTracks`
- Accepts `user_id` and `access_token`.
- Calls helper functions for followed artists and top tracks.
- Converts track data into simplified list (`name`, `url`, `id`).
- Shuffles and returns random 15 tracks as recommendations.

### Lines 528-553: `/recommend`
- Reads `song` query parameter.
- Generates content-based recommendations from local model.
- For each recommendation, calls Spotify search helper for URL/id enrichment.
- Returns enriched results as JSON.

### Lines 555-565: `remove_brackets()`
- Removes bracketed text like `(Remix)` from string using regex.

### Lines 568-607: `/recommendsearch`
- Reads token and query.
- Gets local recommendations first.
- Searches each recommendation on Spotify with cleaned title.
- Returns list with Spotify IDs, names, artists, URLs.

### Lines 608-669: `/search_music`
- Reads token/query/user_id from JSON body.
- Validates required fields.
- Calls Spotify search API with configurable result limit.
- Converts Spotify response into simplified track objects.
- Logs first track interaction for analytics.
- Starts background lyrics scraping thread.
- Returns search results.

### Lines 671-678: `preprocess()`
- Lowercases text, removes brackets/punctuation/digits.
- Removes stopwords and returns cleaned lyric text.

### Lines 680-699: `load_lyrics()`
- Reads all `.txt` files from `lyrics` folder.
- Preprocesses lyrics and extracts title from filename.
- Returns lyric dataframe.
- Loads this dataframe globally into `ldf`.

### Lines 704-714: TF-IDF Load-or-Train
- If saved TF-IDF artifacts exist, load them from disk.
- Else fit TF-IDF on lyrics and save vectorizer + matrix.

### Lines 715-731: `recommend_tfidf()`
- Finds query song index in lyric dataframe.
- Computes cosine similarity against TF-IDF matrix.
- Gets top similar songs.
- Maps similar song titles to Spotify tracks via `search_track()`.

### Lines 736-767: BERT Load + Embeddings
- Loads local SentenceTransformer model if available, else downloads and saves.
- Creates/loads lyric embeddings from `bert_embeddings.npy`.
- Validates embedding count and regenerates if mismatched.
- Attaches embedding vectors to lyric dataframe.

### Lines 769-788: `recommend_bert()`
- Gets query song embedding vector.
- Computes cosine similarity against all BERT embeddings.
- Selects top similar indices and maps to Spotify tracks.

### Lines 790-833: `/getlyricbert`
- Searches Spotify by query.
- Uses first returned track name as seed.
- Runs BERT lyric recommender and returns resulting tracks.

### Lines 834-877: `/getlyrictfidf`
- Same flow as above, but uses TF-IDF lyric recommender.

### Lines 879-904: `search_track()`
- Generic Spotify track search helper (`limit=1`).
- Returns track URL, id, name, artist list (joined string).

### Lines 906-917: `/player/<track_id>`
- Returns inline HTML page with Spotify embedded track player iframe.

### Lines 919-923: `/favicon.ico`
- Serves static favicon from `static/favicon.ico`.

### Lines 925-930: `format_duration()`
- Converts milliseconds into `MM:SS` formatted duration string.

### Lines 933-1047: `get_track_details()`
- Accepts track IDs and token.
- Calls Spotify `/tracks` and `/audio-features`.
- Merges metadata + audio features into structured rows.
- Normalizes release date format.
- Writes result to timestamped CSV file.
- Returns JSON with file info and row data.

### Lines 1050-1060: Commented Genius Code
- Placeholder/commented code for alternate lyrics source (`lyricsgenius`).

### Lines 1062-1127: `get_lyrics_from_lyricsmint()`
- Builds LyricsMint search URL from track + artist.
- Scrapes first matching song page link.
- Fetches lyrics page and extracts lyric paragraphs.
- Saves lyrics into `lyrics/<track>_<artist>.txt`.
- Returns extracted lyrics text.

### Lines 1129-1133: `background_lyrics_fetch()`
- Iterates over track/artist pairs and triggers lyrics scraping.

### Lines 1135-1163: `log_user_interaction()`
- Upserts user-track interaction document in MongoDB.
- Publishes interaction event to Kafka topic (`user-interaction-events`).
- Logs errors safely if any failure occurs.

### Lines 1165-1167: App Entry Point
- If file is run directly, logs startup and launches Flask dev server (`debug=True`).

## `viva.py` Line-by-Line Explanation (In Order)

This file is a refined Flask backend variant of `app.py` with better Spotify error handling, timeout safety, and lyrics/embedding index alignment fixes.

### Lines 1-37: Imports
- Imports core modules, Flask helpers, ML/NLP libraries, lyrics scraping tools, and local helpers.
- Includes duplicate imports (`logging`, `requests`, Flask symbols), but functionality remains unaffected.

### Lines 40-47: Windows UTF-8 Console Safety
- On Windows, reconfigures stdout/stderr to UTF-8.
- Prevents `UnicodeEncodeError` while printing scraped Unicode text.

### Lines 49-53: Unicode Cleanup Helper
- `strip_unicode_format_chars()` removes invisible Unicode format chars (`Cf` category).
- Used before logging/saving text to avoid encoding issues.

### Lines 56-64: Logging + Spotify Limits
- Initializes logger.
- Defines request timeout tuple (`connect`, `read`).
- Defines max Spotify search limit as 10 to prevent Spotify 400 invalid limit errors.

### Lines 66-82: Safe HTTP Wrappers
- `safe_spotify_get()` and `safe_spotify_post()` wrap requests with timeouts and exception handling.
- Return `None` instead of throwing, so routes can return controlled API errors.

### Lines 84-85: Flask App Initialization
- Creates Flask app and enables CORS.

### Lines 87-99: Spotify Error Detail Parser
- Extracts meaningful error message from Spotify JSON or text response.
- Used to provide user-friendly API error messages.

### Lines 101-147: Centralized Spotify Error-to-JSON Mapper
- `jsonify_spotify_search_failure()` maps Spotify status codes to proper response bodies:
  - 401 expired token
  - 403 permission denied
  - 429 rate limit
  - 400 bad input (with limit explanation)
  - other codes mapped to generic upstream error.

### Lines 149-167: Search Param Sanitizer
- `spotify_search_track_params()` validates query and limit.
- Trims query, caps length, converts/caps limit to 1..10.
- Ensures safe and valid Spotify search parameters.

### Lines 170-180: NLP + Paths Setup
- Downloads stopwords and creates stopword set.
- Defines lyrics/saved-model directories and paths for TF-IDF/BERT artifacts.

### Lines 183-193: Spotify OAuth Configuration
- Defines Spotify client credentials and redirect URI (env-overridable).
- Sets OAuth scopes needed by app features.

### Lines 195-214: Base Recommender Artifacts
- Loads `music_recommender.pkl` (cosine similarity + dataframe).
- Loads TF-IDF vectorizer pickle.
- `get_recommendation()` performs content-based top-10 recommendation by cosine similarity.

### Lines 217-220: In-Memory State
- `user_tokens` stores temporary access tokens.
- `user_listening_data` stores fetched user track data.

### Lines 221-227: Basic UI Routes
- `/` serves `index2.html`.
- `/signupindex` serves signup page.

### Lines 229-255: `/signup`
- Reads name/email from JSON.
- Uses email as temporary app user id.
- Stores empty token entry so callback state validation passes.
- Returns Spotify authorization URL and user id.

### Lines 256-277: `/login`
- Generates UUID state/user id.
- Stores placeholder token.
- Redirects to Spotify authorize endpoint with state.

### Lines 278-329: `/callback`
- Validates `code` + `state`.
- Exchanges code for token using safe POST wrapper.
- Fetches Spotify profile using safe GET wrapper.
- Stores token under both OAuth state and Spotify user id.
- Redirects to `/show_token` with token + Spotify user id.

### Lines 330-334: `/show_token`
- Renders `index2.html` and injects token + user id.

### Lines 336-375: `/collect_user_data` (GET/POST)
- Accepts token from POST body or fallback from in-memory map.
- Fetches top tracks via Spotify.
- Builds reduced track features list.
- Stores user listening data and syncs token map.
- Returns success JSON with count.

### Lines 377-413: `/getMyTracks`
- Reads user id + access token.
- Gets user top tracks from helper.
- Validates response and safely extracts fields.
- Randomly samples up to 15 tracks and returns them.

### Lines 414-438: `/recommend`
- Accepts song name and runs local content-based recommendation.
- Enriches each result with Spotify URL/id via `search_track()`.
- Returns enriched list.

### Lines 441-451: `remove_brackets()`
- Removes parenthetical text using regex for cleaner search strings.

### Lines 454-491: `/recommendsearch`
- Gets recommendations from local model for query.
- For each recommendation, performs validated Spotify search (`limit=1`).
- Builds and returns Spotify track list.

### Lines 493-545: `/search_music`
- Validates presence of access token and non-empty query.
- Uses sanitized Spotify params (including limit cap).
- Calls Spotify safely and maps non-200 via centralized handler.
- Returns simplified track list.
- Starts background lyrics fetch thread.

### Lines 548-555: `preprocess()`
- Normalizes lyric text: lowercase, remove brackets/punctuation/digits/stopwords.

### Lines 557-576: `load_lyrics()`
- Reads lyrics text files from `lyrics`.
- Cleans and converts to dataframe with `title` + processed lyrics.
- Loads dataframe globally as `ldf`.

### Lines 578-599: TF-IDF Artifact Validation
- Loads saved TF-IDF artifacts if present.
- If missing or row count mismatches current lyrics set, refits and resaves.

### Lines 600-630: `recommend_tfidf()`
- Uses `ldf_tfidf` alignment-safe index lookup.
- Guards against index mismatch with matrix row count.
- Computes cosine similarity and resolves results to Spotify tracks.

### Lines 631-668: BERT Load and Embedding Alignment
- Loads/downloads SentenceTransformer model.
- Loads/regenerates embeddings based on current valid lyric rows.
- Stores a pre-merge copy (`ldf_tfidf`) to keep TF-IDF row alignment stable.
- Merges BERT vectors into `ldf`.

### Lines 669-694: `recommend_bert()`
- Looks up query title in `valid_df` (the embedding-aligned dataframe).
- Computes cosine similarity over BERT embeddings.
- Returns top similar tracks resolved through Spotify search.

### Lines 696-747: `/getlyricbert`
- Validates search query and Spotify reachability.
- Searches Spotify for seed track list.
- Uses first track name (cleaned) as local lyric key.
- Returns BERT lyric recommendations or meaningful "no local lyrics match" message.

### Lines 748-795: `/getlyrictfidf`
- Same input/search flow as BERT endpoint.
- Runs TF-IDF lyric recommender and returns result tracks.

### Lines 797-820: `search_track()`
- Reusable Spotify single-track search helper with safe request + validated params.
- Returns minimal track metadata dictionary.

### Lines 822-833: `/player/<track_id>`
- Serves HTML page embedding Spotify track player iframe.

### Lines 835-838: `/favicon.ico`
- Serves favicon from static folder.

### Lines 841-846: `format_duration()`
- Converts milliseconds to `MM:SS`.

### Lines 848-967: `get_track_details()`
- For a list of track IDs:
  - Calls Spotify track metadata endpoint.
  - Calls audio-features endpoint.
  - Merges both into structured rows.
  - Writes timestamped CSV (`spotify_hindi_tracks_*.csv`).
- Returns JSON with success message and generated data.

### Lines 970-980: Commented Genius Placeholder
- Legacy commented code for alternate lyrics API (`lyricsgenius`).

### Lines 982-1041: `get_lyrics_from_lyricsmint()`
- Searches LyricsMint page for song.
- Fetches first song lyrics page.
- Extracts lyrics text from specific HTML container.
- Removes invisible Unicode format chars.
- Saves lyrics file in `lyrics/<track>_<artist>.txt`.

### Lines 1043-1051: `background_lyrics_fetch()`
- Iterates over tracks/artists in background.
- Logs progress and isolates per-track exceptions so thread continues.

### Lines 1054-1056: Entry Point
- Logs server start and runs Flask app in debug mode.

## 20 Likely Viva Questions with Ideal Answers

1. **What problem does your project solve?**  
   It recommends songs personalized to a user by combining content similarity, collaborative behavior, and lyric semantics.

2. **Why did you choose Flask?**  
   Flask is lightweight, fast to prototype APIs, and easy to integrate with ML models and external APIs like Spotify.

3. **How does OAuth work in your app?**  
   User is redirected to Spotify authorization, app receives `code` in callback, exchanges it for access token, then calls Spotify APIs.

4. **What is your main recommendation function?**  
   `get_recommendation()` for content-based similarity and `hybrid_recommendations()` for combining collaborative + content outputs.

5. **How does content-based recommendation work here?**  
   It uses precomputed cosine similarity matrix over song features and returns top-N nearest songs.

6. **How does collaborative filtering work here?**  
   Spark ALS is trained on user-track interaction scores from MongoDB and predicts user-specific items.

7. **What is the role of Redis in your app?**  
   Stores recent recommendation responses (`setex`) so repeated requests are served quickly.

8. **What is the purpose of Kafka here?**  
   Streams user interaction/listening events asynchronously for real-time analytics and model updates.

9. **Why MongoDB and not SQL?**  
   Interaction/event data is semi-structured and evolves; Mongo gives schema flexibility and easy upserts.

10. **What is the difference between TF-IDF and BERT in your project?**  
    TF-IDF matches word overlap; BERT captures contextual meaning in lyrics.

11. **How do you preprocess lyrics?**  
    Lowercasing, removing brackets/punctuation/digits, stopword removal, then vectorization/embedding.

12. **How do you handle service unavailability?**  
    `init_big_data()` falls back to local mode (`None`/noop Redis), keeping app functional without full big-data stack.

13. **What is cold-start and how do you address it?**  
    Cold-start is lack of user interaction history; content-based and popularity/top-track methods still provide recommendations.

14. **How do you reduce API latency?**  
    Redis caching, pre-saved models, and asynchronous/background operations like lyrics fetch threads.

15. **How do you log and trace API calls?**  
    `before_request` and `after_request` middleware adds request IDs, logs method/path/status, and measures duration.

16. **How do you store user interactions?**  
    Upsert in MongoDB with incrementing counts and timestamp; optionally publish to Kafka topics.

17. **How is hybrid recommendation formed currently?**  
    Simple merge split between collaborative and content-based lists (demo strategy), then cached.

18. **What are current limitations of your implementation?**  
    Hardcoded secrets/tokens in code, duplicate initializations, and simplistic merge strategy that can be improved.

19. **How would you improve recommendation quality?**  
    Weighted rank fusion, deduplication, context-aware features (time/mood), and online feedback loops.

20. **How would you scale this to production?**  
    Move secrets to env vault, containerize services, deploy managed Kafka/Mongo/Redis, periodic Spark retraining, monitoring + CI/CD.

## Short Closing Script for Viva

"Our system is a hybrid recommender built with Flask and Spotify integration.  
We combine content-based similarity, collaborative filtering through Spark ALS, and lyric semantics via TF-IDF/BERT.  
Kafka handles event streaming, Mongo stores interactions, and Redis optimizes latency through caching.  
This design balances recommendation quality, scalability, and response speed."

