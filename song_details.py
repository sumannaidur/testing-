import os
import time
import spotipy
import pandas as pd
import logging
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime  # ✅ FIXED: Import datetime

# Load API credentials from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, filename="spotify_scraper.log", filemode="a", format="%(asctime)s - %(message)s")

# Rate limit variables
REQUESTS_MADE = 0
START_TIME = time.time()

# Spotify API credentials list
SPOTIFY_CREDENTIALS = [
    {"client_id": os.getenv("SPOTIFY_CLIENT_ID_1"), "client_secret": os.getenv("SPOTIFY_CLIENT_SECRET_1")},
    {"client_id": os.getenv("SPOTIFY_CLIENT_ID_2"), "client_secret": os.getenv("SPOTIFY_CLIENT_SECRET_2")},
    {"client_id": os.getenv("SPOTIFY_CLIENT_ID_3"), "client_secret": os.getenv("SPOTIFY_CLIENT_SECRET_3")}
]

credential_index = 0  # Track which API key is in use
sp = None  # Global Spotify client instance

# Initialize Spotify Client
def get_spotify_client():
    creds = SPOTIFY_CREDENTIALS[credential_index]
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=creds["client_id"], client_secret=creds["client_secret"]))

# Initialize the first client
sp = get_spotify_client()

# Rate Limiter
def rate_limiter():
    global REQUESTS_MADE, START_TIME
    REQUESTS_MADE += 1
    elapsed_time = time.time() - START_TIME

    if REQUESTS_MADE >= 175:  # Buffer before hitting 180 limit
        sleep_time = 60 - elapsed_time  # Wait for 60 sec rolling window
        if sleep_time > 0:
            logging.info(f"Rate limit nearing! Sleeping for {round(sleep_time, 2)} seconds...")
            time.sleep(sleep_time)
        
        # Reset counter after sleep
        REQUESTS_MADE = 0
        START_TIME = time.time()

# Switch Spotify Credentials
def switch_spotify_client():
    global credential_index, sp
    logging.warning("Rate limit reached! Retrying in 10 seconds before switching credentials...")
    time.sleep(10)  # Avoid immediate bans before switching

    credential_index = (credential_index + 1) % len(SPOTIFY_CREDENTIALS)
    sp = get_spotify_client()
    logging.info(f"Switched to Spotify credentials set {credential_index + 1}")

# Search for a movie album on Spotify
def search_album_on_spotify(movie_name, year, language):
    """Search for the movie album on Spotify."""
    try:
        rate_limiter()
        search_query = f"{movie_name} {year} {language} soundtrack"
        results = sp.search(q=search_query, type="album", limit=5)

        if results["albums"]["items"]:
            for album in results["albums"]["items"]:
                album_name = album["name"].lower()
                if movie_name.lower() in album_name and str(year) in album_name:
                    return album["id"]
            return results["albums"]["items"][0]["id"]

    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 429:
            switch_spotify_client()
            return search_album_on_spotify(movie_name, year, language)
        else:
            logging.error(f"Spotify API error for {movie_name}: {e}")

    return None

# Fetch multiple track details efficiently
def get_songs_from_album(album_id, movie_name):
    """Fetch multiple song details efficiently."""
    songs_data = []

    if album_id:
        try:
            rate_limiter()
            tracks = sp.album_tracks(album_id)
            track_ids = [track["id"] for track in tracks["items"]]

            for i in range(0, len(track_ids), 50):
                rate_limiter()
                track_details = sp.tracks(track_ids[i:i+50])["tracks"]

                for song_meta in track_details:
                    songs_data.append({
                        "Movie": movie_name,
                        "Song Name": song_meta["name"],
                        "Album": song_meta["album"]["name"],
                        "Artist": ", ".join([artist["name"] for artist in song_meta["artists"]]),
                        "Spotify ID": song_meta["id"],
                        "Release Date": song_meta["album"]["release_date"],
                        "Popularity": song_meta["popularity"]
                    })
        except Exception as e:
            logging.error(f"Error fetching songs from album {album_id}: {e}")

    return songs_data

# Save progress
def save_progress(processed_movies, filename="processed_movies.txt"):
    """Save processed movies to a file to avoid duplicate processing."""
    with open(filename, "w") as file:
        file.write("\n".join(processed_movies))

# Process a single CSV file
def process_movie_csv(file_path, processed_movies):
    """Process each movie CSV file and collect song details."""
    print(f"📂 Processing file: {file_path}")  # NEW DEBUG PRINT
    
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    for _, row in df.iterrows():
        movie_name = row["Title"]
        print(f"🎬 Processing movie: {movie_name}")  # NEW DEBUG PRINT

        if movie_name in processed_movies:
            print(f"⏩ Skipping {movie_name} (already processed)")
            continue

        try:
            year = datetime.strptime(str(row["Release Date"]), "%d-%m-%Y").year
        except ValueError:
            print(f"⚠️ Invalid date format for movie: {movie_name}")
            continue

        language = row["Language"]
        print(f"🔍 Searching for: {movie_name} ({year}, {language})")  # NEW DEBUG PRINT

        album_id = search_album_on_spotify(movie_name, year, language)

        if album_id:
            print(f"✅ Album found for {movie_name} - ID: {album_id}")  # NEW DEBUG PRINT
            songs = get_songs_from_album(album_id, movie_name)
            if songs:
                print(f"🎵 Found {len(songs)} songs for {movie_name}")
            else:
                print(f"❌ No songs found for {movie_name}")
        else:
            print(f"❌ No album found for {movie_name}")

        processed_movies.add(movie_name)
        save_progress(processed_movies)  # ✅ FIXED: Now properly defined

# Process all CSV files in parallel
def process_all_csv_files():
    MOVIE_CSV_FOLDER = "movie_csvs"
    csv_files = [os.path.join(MOVIE_CSV_FOLDER, f) for f in os.listdir(MOVIE_CSV_FOLDER) if f.endswith(".csv")]

    if not csv_files:
        print("❌ No CSV files found in 'movie_csvs' folder!")
        return

    print(f"✅ Found {len(csv_files)} CSV files. Processing now...")

    processed_movies = set()
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(process_movie_csv, csv_files, [processed_movies] * len(csv_files))

if __name__ == "__main__":
    print("🚀 Starting Spotify Data Extraction...")
    process_all_csv_files()
    print("🎉 Done!") 
