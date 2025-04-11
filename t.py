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
        
        y, sr = librosa.load(file_path, sr=None)  # Load with original sample rate
        
        # Compute features
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)

        # Convert loudness to dB scale (Spotify uses -60 to 0 dB)
        loudness = np.mean(librosa.amplitude_to_db(rms))  
        loudness = max(-60, min(0, loudness))  # Clamp between -60 and 0 dB

        # Normalize values where needed
        def normalize(value, min_val, max_val):
            return max(0.0, min(1.0, (value - min_val) / (max_val - min_val))) if max_val > min_val else 0.0

        features = {
            "duration_ms": int(duration * 1000),
            "tempo": float(tempo),  # Ensure it's a single float
            "danceability": normalize(np.std(np.diff(librosa.feature.tempogram(y=y, sr=sr))), 0, 0.5),
            "acousticness": normalize(np.mean(zcr), 0, 0.2),  # Normalize ZCR
            "energy": normalize(np.mean(rms), 0, 0.1),  # Normalize RMS
            "instrumentalness": normalize(np.mean(harmonic) / (np.mean(percussive) + 1e-6), 0, 10),
            "liveness": normalize(np.mean(spec_contrast), 0, 2),
            "loudness": loudness,  # Already in correct range
            "speechiness": normalize(np.var(chroma), 0, 1.0),  # Fixed normalization
            "valence": normalize(np.mean(spectral_centroid) / (np.max(spectral_centroid) + 1e-6), 0, 1),  
            "key": int(np.argmax(np.mean(chroma, axis=1))),  
            "mode": int(np.argmax(np.mean(chroma, axis=1)) in [0, 2, 4, 5, 7, 9, 11]),  
            "time_signature": min(max(3, len(librosa.onset.onset_detect(y=y, sr=sr))), 7)  
        }

        return features
    except Exception as e:
        print(f"❌ Error extracting audio features: {e}")
        return None




def search_album_on_spotify(movie_name, year, language, hero=None, heroine=None):
    try:
        rate_limiter()
        search_terms = f"{movie_name} {year} {language} soundtrack"
        
        if hero:
            search_terms += f" {hero}"
        if heroine:
            search_terms += f" {heroine}"
        
        results = sp.search(q=search_terms, type="album", limit=5)
        if results["albums"]["items"]:
            for album in results["albums"]["items"]:
                album_name = album["name"].lower()
                if movie_name.lower() in album_name and str(year) in album_name:
                    return album["id"]
            return results["albums"]["items"][0]["id"]
    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 429:
            switch_spotify_client()
            return search_album_on_spotify(movie_name, year, language, hero, heroine)
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
        file.write("\n".join(processed_movies))

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

            # Create folder path: output/<language>/<year>
            folder_path = os.path.join("output", sanitize_filename(language), str(year))
            os.makedirs(folder_path, exist_ok=True)

            # Save all songs of the year+language to a single CSV
            output_file = os.path.join(folder_path, "songs.csv")

            if os.path.exists(output_file):
                existing_df = pd.read_csv(output_file)
                output_df = pd.concat([existing_df, output_df], ignore_index=True)

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