import time
import spotipy
import pandas as pd
import logging
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from datetime import datetime
from yt_dlp import YoutubeDL
import librosa
import numpy as np
from tqdm import tqdm
import os

# Load API credentials
load_dotenv()

logging.basicConfig(level=logging.INFO, filename="spotify_scraper.log", filemode="a", format="%(asctime)s - %(message)s")

REQUESTS_MADE = 0
START_TIME = time.time()

SPOTIFY_CREDENTIALS = [
    {"client_id": os.getenv("SPOTIFY_CLIENT_ID_1"), "client_secret": os.getenv("SPOTIFY_CLIENT_SECRET_1")},
    {"client_id": os.getenv("SPOTIFY_CLIENT_ID_2"), "client_secret": os.getenv("SPOTIFY_CLIENT_SECRET_2")},
    {"client_id": os.getenv("SPOTIFY_CLIENT_ID_3"), "client_secret": os.getenv("SPOTIFY_CLIENT_SECRET_3")}
]

credential_index = 0
sp = None

def get_spotify_client():
    creds = SPOTIFY_CREDENTIALS[credential_index]
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=creds["client_id"], client_secret=creds["client_secret"]))

sp = get_spotify_client()

def rate_limiter():
    global REQUESTS_MADE, START_TIME
    REQUESTS_MADE += 1
    elapsed_time = time.time() - START_TIME
    if REQUESTS_MADE >= 175:
        sleep_time = 60 - elapsed_time
        if sleep_time > 0:
            logging.info(f"Rate limit nearing! Sleeping for {round(sleep_time, 2)} seconds...")
            time.sleep(sleep_time)
        REQUESTS_MADE = 0
        START_TIME = time.time()

def switch_spotify_client():
    global credential_index, sp
    logging.warning("Rate limit reached! Retrying in 10 seconds before switching credentials...")
    time.sleep(10)
    credential_index = (credential_index + 1) % len(SPOTIFY_CREDENTIALS)
    sp = get_spotify_client()
    logging.info(f"Switched to Spotify credentials set {credential_index + 1}")

def sanitize_filename(name):
    import re
    return re.sub(r'[\\\\/*?:"<>|]', "", name)

def download_song_youtube(song_name, artist_name, save_path="downloads"):
    os.makedirs(save_path, exist_ok=True)
    query = f"{song_name} {artist_name} audio"
    filename = f"{sanitize_filename(song_name)} - {sanitize_filename(artist_name)}.%(ext)s"
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'quiet': True,
        'outtmpl': os.path.join(save_path, filename),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([f"ytsearch1:{query}"])
            return os.path.join(save_path, filename.replace("%(ext)s", "mp3"))
        except Exception as e:
            print(f"❌ Failed to download {song_name} - {artist_name}: {e}")
            return None

def extract_audio_features(file_path):
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 10000:
            raise Exception("File is invalid or too small")
        y, sr = librosa.load(file_path)
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features = {
            "length": duration,
            "tempo": tempo,
            "danceability": np.mean(librosa.feature.tempogram(y=y, sr=sr)),
            "acousticness": np.mean(librosa.feature.rms(y=y)),
            "energy": np.mean(librosa.feature.melspectrogram(y=y, sr=sr)),
            "instrumentalness": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            "liveness": np.mean(librosa.feature.spectral_flatness(y=y)),
            "valence": np.mean(librosa.feature.zero_crossing_rate(y)),
            "loudness": np.mean(librosa.feature.rms(y=y)),
            "speechiness": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            "key": np.argmax(np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)),
            "time_signature": 4
        }
        return features
    except Exception as e:
        print(f"❌ Error extracting audio features: {e}")
        return None

def search_album_on_spotify(movie_name, year, language):
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

def get_songs_from_album(album_id, movie_name):
    songs_data = []
    if album_id:
        try:
            rate_limiter()
            tracks = sp.album_tracks(album_id)
            track_ids = [track["id"] for track in tracks["items"]]
            for i in tqdm(range(0, len(track_ids), 50), desc=f"🎵 {movie_name}", unit="batch"):
                rate_limiter()
                track_details = sp.tracks(track_ids[i:i+50])["tracks"]
                for song_meta in track_details:
                    audio_file = download_song_youtube(song_meta["name"], song_meta["artists"][0]["name"])
                    if audio_file:
                        audio_features = extract_audio_features(audio_file)
                        if audio_features:
                            songs_data.append({
                                "name": song_meta["name"],
                                "album": song_meta["album"]["name"],
                                "artist": ", ".join([artist["name"] for artist in song_meta["artists"]]),
                                "id": song_meta["id"],
                                "release_date": song_meta["album"]["release_date"],
                                "popularity": song_meta["popularity"],
                                **audio_features
                            })
                        try:
                            os.remove(audio_file)
                        except Exception as e:
                            logging.warning(f"Could not delete file {audio_file}: {e}")
        except Exception as e:
            logging.error(f"Error fetching songs from album {album_id}: {e}")
    return songs_data

def save_progress(processed_movies, filename="processed_movies.txt"):
    with open(filename, "w") as file:
        file.write("\\n".join(processed_movies))

def process_movie_csv(file_path, processed_movies):
    print(f"📂 Processing file: {file_path}")
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"🎬 {os.path.basename(file_path)}", unit="movie"):
        movie_name = row["Title"]
        if movie_name in processed_movies:
            continue
        try:
            year = datetime.strptime(str(row["Release Date"]), "%d-%m-%Y").year
        except ValueError:
            continue
        language = row["Language"]
        album_id = search_album_on_spotify(movie_name, year, language)
        songs = get_songs_from_album(album_id, movie_name)
        if songs:
            output_df = pd.DataFrame(songs)
            output_file = f"output/{sanitize_filename(movie_name.replace(' ', '_'))}_songs.csv"
            os.makedirs("output", exist_ok=True)
            output_df.to_csv(output_file, index=False)
        processed_movies.add(movie_name)
        save_progress(processed_movies)

def process_all_csv_files():
    MOVIE_CSV_FOLDER = "movie_csvs"
    csv_files = [os.path.join(MOVIE_CSV_FOLDER, f) for f in os.listdir(MOVIE_CSV_FOLDER) if f.endswith(".csv")]
    if not csv_files:
        print("❌ No CSV files found in 'movie_csvs' folder!")
        return
    print(f"✅ Found {len(csv_files)} CSV files. Processing now...")
    processed_movies = set()
    for file in tqdm(csv_files, desc="📁 CSV Files", unit="file"):
        process_movie_csv(file, processed_movies)

if __name__ == "__main__":
    print("🚀 Starting Spotify + YouTube Audio Feature Extraction...")
    process_all_csv_files()
    print("🎉 Done!")