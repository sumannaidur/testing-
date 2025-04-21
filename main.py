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
from pydub import AudioSegment
from filelock import FileLock
import lyricsgenius
from difflib import SequenceMatcher

# Load API credentials
load_dotenv()

GENIUS_TOKEN = "VlwXDvo5O8sybpEZ2Vfn8DCnDthcgs8IghLHPmX7I0FUshz6fwh0yIYYBF5hN7-3"  # Replace with your Genius token
genius = lyricsgenius.Genius(GENIUS_TOKEN, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"])
logging.basicConfig(
    level=logging.DEBUG,  # <- changed from INFO to DEBUG
    filename="spotify_scraper.log",filemode="a",format="%(asctime)s - %(levelname)s - %(message)s"
)

REQUESTS_MADE = 0
START_TIME = time.time()
SPOTIFY_CREDENTIALS = [
    {"client_id": os.getenv("SPOTIFY_CLIENT_ID_1"), "client_secret": os.getenv("SPOTIFY_CLIENT_SECRET_1")},
    {"client_id": os.getenv("SPOTIFY_CLIENT_ID_2"), "client_secret": os.getenv("SPOTIFY_CLIENT_SECRET_2")},
    {"client_id": os.getenv("SPOTIFY_CLIENT_ID_3"), "client_secret": os.getenv("SPOTIFY_CLIENT_SECRET_3")}
]
credential_index = 0
sp = None

def cleanup_files(file_list):
    for file in file_list:
        try:
            if file and os.path.exists(file):
                os.remove(file)
                logging.debug(f"🧹 Deleted: {file}")
        except Exception as e:
            logging.warning(f"⚠ Could not delete file {file}: {e}")

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
    sanitized = re.sub(r'[\\\\/*?:"<>|]', "", name)
    logging.debug(f"Sanitized filename: {name} -> {sanitized}")
    return sanitized

def download_song_youtube(song_name, artist_name, save_path="downloads", cookie_path="youtube_cookies.txt"):
    os.makedirs(save_path, exist_ok=True)
    query = f"{song_name} {artist_name} audio"
    sanitized_song = sanitize_filename(song_name)
    sanitized_artist = sanitize_filename(artist_name)
    filename = f"{sanitized_song} - {sanitized_artist}.%(ext)s"
    full_output_path = os.path.join(save_path, filename.replace("%(ext)s", "mp3"))
    print(f"🔍 Attempting to download: {song_name} by {artist_name}")
    logging.info(f"Downloading YouTube audio: Query='{query}' | Output='{full_output_path}'")
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'quiet': True,
        'cookies': 'youtube_cookies.txt',
        'outtmpl': os.path.join(save_path, filename),
        'postprocessors': [{'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3',
                            'preferredquality': '192',
                            }],
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"ytsearch1:{query}"])
        if os.path.exists(full_output_path):
            print(f"✅ Downloaded and converted: {full_output_path}")
            logging.info(f"Download successful: {full_output_path}")
            return full_output_path
        else:
            print(f"❌ File not found after download: {full_output_path}")
            logging.warning(f"File missing after YouTubeDL process: {full_output_path}")
            return None
    except Exception as e:
        print(f"❌ Failed to download {song_name} - {artist_name}: {e}")
        logging.error(f"Download failed for {song_name} - {artist_name}: {e}")
        return None
def convert_mp3_to_wav(mp3_path):
    wav_path = mp3_path.replace(".mp3", ".wav")
    print(f"🔄 Converting MP3 to WAV: {mp3_path} ➡ {wav_path}")
    logging.info(f"Converting MP3 to WAV: {mp3_path} -> {wav_path}")   
    try:
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(wav_path, format="wav")
        if os.path.exists(wav_path):
            print(f"✅ Successfully converted to WAV: {wav_path}")
            logging.info(f"MP3 to WAV conversion successful: {wav_path}")
            return wav_path
        else:
            print(f"❌ Conversion failed, WAV file not found: {wav_path}")
            logging.warning(f"WAV file not created after conversion: {wav_path}")
            return None
    except Exception as e:
        print(f"❌ Failed to convert {mp3_path} to WAV: {e}")
        logging.error(f"Error converting {mp3_path} to WAV: {e}")
        return None
def extract_audio_features(file_path):
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 10000:
            raise Exception("File is invalid or too small")
        print(f"🎧 Extracting features from: {file_path}")
        y, sr = librosa.load(file_path, sr=None, mono=True, duration=60)
        # Optimized parameters
        hop_length = 512
        n_fft = 2048
        # Core features
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        tempogram = librosa.feature.tempogram(y=y, sr=sr, hop_length=hop_length)
        loudness = np.mean(librosa.amplitude_to_db(rms))
        loudness = max(-60, min(0, loudness))  # Clamp to Spotify's range
        # Helper to normalize values
        def normalize(value, min_val, max_val):
            return max(0.0, min(1.0, (value - min_val) / (max_val - min_val))) if max_val > min_val else 0.0
        # Precomputed means to avoid duplication
        mean_chroma = np.mean(chroma, axis=1)
        mean_spec_contrast = np.mean(spec_contrast)
        mean_rms = np.mean(rms)
        mean_zcr = np.mean(zcr)
        mean_harmonic = np.mean(harmonic)
        mean_percussive = np.mean(percussive)
        var_chroma = np.var(chroma)
        mean_spec_centroid = np.mean(spectral_centroid)
        max_spec_centroid = np.max(spectral_centroid)
        features = {
            "duration_ms": int(duration * 1000),
            "tempo": float(tempo.item()) if hasattr(tempo, 'item') else float(tempo),
            "danceability": normalize(np.std(np.diff(tempogram)), 0, 0.5),
            "acousticness": normalize(mean_zcr, 0, 0.2),
            "energy": normalize(mean_rms, 0, 0.1),
            "instrumentalness": normalize(mean_harmonic / (mean_percussive + 1e-6), 0, 10),
            "liveness": normalize(mean_spec_contrast, 0, 2),
            "loudness": loudness,
            "speechiness": normalize(var_chroma, 0, 1.0),
            "valence": normalize(mean_spec_centroid / (max_spec_centroid + 1e-6), 0, 1),
            "key": int(np.argmax(mean_chroma)),
            "mode": int(np.argmax(mean_chroma) in [0, 2, 4, 5, 7, 9, 11]),
            "time_signature": min(max(3, len(librosa.onset.onset_detect(y=y, sr=sr))), 7)
        }
        print(f"✅ Features extracted successfully for {os.path.basename(file_path)}")
        return features
    except Exception as e:
        print(f"❌ Error extracting audio features from {file_path}: {e}")
        return None

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def search_album_on_spotify(movie_name, year, language, music_director=None, hero=None):
    try:
        rate_limiter()
        
        def clean(name):
            return name.lower().replace("-", " ").replace("_", " ").strip() if name else ""
        
        movie_lower = clean(movie_name)
        year_str = str(year)
        director_lower = clean(music_director)
        hero_lower = clean(hero)
        
        def search_with_query(artist_name=None):
            query = f'{movie_name} {year} {language}'
            if artist_name:
                query += f' {artist_name}'
            print(f"🔍 Spotify Search Query: {query}")
            logging.info(f"🔍 Searching Spotify for: {query}")
            return sp.search(q=query, type="album", limit=10)
        
        # Try: 1) music director, 2) hero, 3) no artist
        search_attempts = [music_director, hero, None]
        for artist in search_attempts:
            results = search_with_query(artist)
            if results["albums"]["items"]:
                break  # Got some albums
        else:
            logging.warning(f"❌ No albums found for: {movie_name} ({year}) in any attempt.")
            return None
        
        # Matching logic
        for album in results["albums"]["items"]:
            album_name = clean(album["name"])
            artist_names = " ".join([clean(artist["name"]) for artist in album["artists"]])
            if movie_lower in album_name and (director_lower in artist_names or hero_lower in artist_names):
                print(f"✅ Strong match: {album['name']} by {artist_names}")
                logging.info(f"✅ Strong match: {album['name']}")
                return album["id"]
        
        for album in results["albums"]["items"]:
            album_name = clean(album["name"])
            if movie_lower in album_name and year_str in album_name:
                print(f"✅ Medium match (movie + year): {album['name']}")
                logging.info(f"✅ Medium match: {album['name']}")
                return album["id"]
        
        for album in results["albums"]["items"]:
            album_name = clean(album["name"])
            if movie_lower in album_name:
                print(f"⚠️ Partial match (movie only): {album['name']}")
                logging.info(f"⚠️ Partial match: {album['name']}")
                return album["id"]
        
        # Fallback using similarity
        best_match = None
        highest_score = 0
        for album in results["albums"]["items"]:
            score = similarity(movie_lower, clean(album["name"]))
            if score > highest_score:
                highest_score = score
                best_match = album

        if best_match:
            print(f"🚨 Fallback (best similarity {highest_score:.2f}): {best_match['name']}")
            logging.warning(f"⚠ No strong match. Returning best similar album: {best_match['name']}")
            return best_match["id"]
        else:
            logging.error("❌ No suitable fallback album found.")
            return None
    
    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 429:
            logging.warning("⏳ Rate limit hit, switching Spotify client...")
            switch_spotify_client()
            return search_album_on_spotify(movie_name, year, language, music_director, hero)
        else:
            logging.error(f"❌ Spotify API error for {movie_name}: {e}")
            return None
    except Exception as e:
        logging.error(f"❌ Unexpected error in search_album_on_spotify: {e}")
        return None

def sanitize_filename(name):
    # Remove characters that are invalid in filenames
    return "".join(c for c in name if c.isalnum() or c in " _-").rstrip()

def save_lyrics(lyrics, album_name, song_name):
    try:
        folder_name = os.path.join("lyrics", sanitize_filename(album_name))
        os.makedirs(folder_name, exist_ok=True)
        file_path = os.path.join(folder_name, f"{sanitize_filename(song_name)}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(lyrics)
        logging.info(f"💾 Saved lyrics: {file_path}")
    except Exception as e:
        logging.error(f"❌ Failed to save lyrics for {song_name}: {e}")

def fetch_lyrics_from_genius(song_name, artist_name, album_name):
    try:
        song = genius.search_song(title=song_name, artist=artist_name)
        if song and song.lyrics:
            logging.info("📜 Got lyrics")
            save_lyrics(song.lyrics, album_name, song_name)
            return song.lyrics
        return ""
    except Exception as e:
        logging.warning(f"⚠ Genius API failed for {song_name} by {artist_name}: {e}")
        return ""

def get_songs_from_album(album_id, movie_name):
    songs_data = []
    if not album_id:
        logging.warning(f"⚠ Skipping: No album ID for '{movie_name}'")
        return songs_data
    try:
        rate_limiter()
        tracks = sp.album_tracks(album_id)
        track_ids = [track["id"] for track in tracks["items"]]
        for i in tqdm(range(0, len(track_ids), 50), desc=f"🎵 {movie_name}", unit="batch"):
            rate_limiter()
            track_details = sp.tracks(track_ids[i:i+50])["tracks"]
            for song_meta in track_details:
                song_name = song_meta["name"]
                artist_name = song_meta["artists"][0]["name"]
                logging.info(f"🎧 Processing: {song_name} by {artist_name}")
                audio_file = download_song_youtube(song_name, artist_name)
                if not audio_file:
                    logging.warning(f"❌ Skipping download failure: {song_name}")
                    continue
                wav_file = convert_mp3_to_wav(audio_file)
                if not wav_file:
                    logging.warning(f"❌ Skipping conversion failure: {song_name}")
                    cleanup_files([audio_file])
                    continue
                audio_features = extract_audio_features(wav_file)
                if not audio_features:
                    logging.warning(f"❌ Skipping feature extraction failure: {song_name}")
                    cleanup_files([audio_file, wav_file])
                    continue
                fetch_lyrics_from_genius(song_name, artist_name, song_meta["album"]["name"])
                songs_data.append({
                    "name": song_name,
                    "album": song_meta["album"]["name"],
                    "artist": ", ".join([artist["name"] for artist in song_meta["artists"]]),
                    "id": song_meta["id"],
                    "release_date": song_meta["album"]["release_date"],
                    "popularity": song_meta["popularity"],
                    **audio_features
                })
                cleanup_files([audio_file, wav_file])
    except Exception as e:
        logging.error(f"🔥 Error processing album {album_id} ({movie_name}): {e}")
    return songs_data

def save_progress(processed_movies, filename="processed_movies.txt"):
    try:
        with FileLock(filename + ".lock"):
            with open(filename, "w") as file:
                file.write("\n".join(sorted(set(processed_movies))))
        print(f"💾 Progress saved. Total processed movies: {len(processed_movies)}")
    except Exception as e:
        print(f"❌ Failed to save progress to {filename}: {e}")

def load_processed_movies(filename="processed_movies.txt"):
    try:
        if not os.path.exists(filename):
            return set()
        with open(filename, "r") as file:
            processed = {line.strip() for line in file if line.strip()}
        print(f"📂 Loaded {len(processed)} processed movies from {filename}")
        return processed
    except Exception as e:
        print(f"❌ Error loading processed movies: {e}")
        return set()

def process_movie_csv(file_path, processed_movies):
    print(f"📂 Processing file: {file_path}")
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"🎬 {os.path.basename(file_path)}", unit="movie"):
        movie_name = row.get("Title", "").strip()
        if not movie_name or movie_name in processed_movies:
            continue
        try:
            release_date = row.get("Release Date", "")
            year = datetime.strptime(str(release_date), "%d-%m-%Y").year
            language = row.get("Language", "").strip()
            hero = row.get("Hero", "").strip()
            music_director = row.get("Music Director", "").strip()
            album_id = search_album_on_spotify(movie_name, year, language, music_director, hero)
            songs = get_songs_from_album(album_id, movie_name)
            if songs:
                output_df = pd.DataFrame(songs)
                folder_path = os.path.join("output", sanitize_filename(language), str(year))
                os.makedirs(folder_path, exist_ok=True)
                output_file = os.path.join(folder_path, "songs.csv")
                if os.path.exists(output_file):
                    existing_df = pd.read_csv(output_file)
                    output_df = pd.concat([existing_df, output_df], ignore_index=True)
                output_df.to_csv(output_file, index=False)
                print(f"✅ Saved {len(songs)} songs for '{movie_name}' → {output_file}")
            else:
                print(f"⚠ No songs found for: {movie_name}")
        except Exception as e:
            print(f"❌ Error processing movie '{movie_name}': {e}")
        processed_movies.add(movie_name)
        save_progress(processed_movies)

def process_all_csv_files():
    MOVIE_CSV_FOLDER = "movie_csvs"
    if not os.path.exists(MOVIE_CSV_FOLDER):
        print(f"❌ Folder '{MOVIE_CSV_FOLDER}' does not exist!")
        return
    # Recursively gather all CSV files in subfolders
    csv_files = []
    for root, _, files in os.walk(MOVIE_CSV_FOLDER):
        for file in files:
            if file.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    if not csv_files:
        print("❌ No CSV files found in 'movie_csvs' or subfolders!")
        return
    print(f"✅ Found {len(csv_files)} CSV files across language folders. Processing now...")
    processed_movies = load_processed_movies()
    for file in tqdm(csv_files, desc="📁 CSV Files", unit="file"):
        try:
            process_movie_csv(file, processed_movies)
        except Exception as e:
            print(f"⚠ Error processing file {file}: {e}")
    print("🎉 All CSV files processed successfully.")

if __name__ == "__main__":
    print("🚀 Starting Spotify + YouTube Audio Feature Extraction...")
    process_all_csv_files()
    print("🎉 Done!")