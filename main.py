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
import re
import google.generativeai as genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

# Load API credentials
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY")) 
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
    sanitized = re.sub(r'[\\\\/*?:"<>|]', "", name)
    logging.debug(f"Sanitized filename: {name} -> {sanitized}")
    return sanitized

import webbrowser

def download_song_youtube(song_name, artist_name, save_path="downloads", cookie_path="youtube_cookies.txt"):
    global REQUESTS_MADE  # To ensure rate limiting works globally
    os.makedirs(save_path, exist_ok=True)
    query = f"{song_name} {artist_name} audio"
    sanitized_song = sanitize_filename(song_name)
    sanitized_artist = sanitize_filename(artist_name)
    filename = f"{sanitized_song} - {sanitized_artist}.%(ext)s"
    full_output_path = os.path.join(save_path, filename.replace("%(ext)s", "mp3"))

    print(f"🔍 Attempting to download: {song_name} by {artist_name}")
    logging.info(f"Downloading YouTube audio: Query='{query}' | Output='{full_output_path}'")

    # Call your custom rate limiter here
    rate_limiter()

    # Check cookie file
    if not os.path.exists(cookie_path):
        print(f"⚠️ Warning: Cookie file not found → {cookie_path}")
        logging.warning(f"Cookie file not found: {cookie_path}")
    else:
        print(f"🍪 Using cookies file: {cookie_path}")
        logging.info(f"Using cookies from: {cookie_path}")

    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'quiet': True,
        'cookies': cookie_path,
        'outtmpl': os.path.join(save_path, filename),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
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
        err_msg = str(e)
        print(f"❌ Failed to download {song_name} - {artist_name}: {err_msg}")
        logging.error(f"Download failed for {song_name} - {artist_name}: {err_msg}")

        # Detect YouTube login prompt
        if "Sign in to confirm" in err_msg:
            print("🔐 YouTube is asking for login. Opening browser...")
            search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            webbrowser.open(search_url)
            input("⏳ After logging in and verifying, press Enter here to retry download...")
            return download_song_youtube(song_name, artist_name, save_path, cookie_path)

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

from difflib import SequenceMatcher

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def clean(text):
    return text.lower().replace("-", " ").replace("_", " ").strip() if text else ""

def match_album_heuristic(album, movie_name, year, language, music_director, hero):
    album_name = clean(album["name"])
    movie_name = clean(movie_name)
    name_score = similarity(album_name, movie_name)

    artist_names = " ".join([clean(a["name"]) for a in album["artists"]])
    artist_match = music_director and clean(music_director) in artist_names
    hero_match = hero and clean(hero) in artist_names
    year_match = str(year) in album.get("release_date", "")
    lang_match = clean(language) in album_name or clean(language) in artist_names

    score = 0
    if name_score >= 0.85:
        score += 2
    elif name_score >= 0.75:
        score += 1

    if artist_match or hero_match:
        score += 1
    if year_match:
        score += 1
    if lang_match:
        score += 1

    return score, name_score

def search_album_on_spotify(movie_name, year, language, music_director=None, hero=None, market="IN"):
    try:
        rate_limiter()

        search_phrases = [
            f"{movie_name} {year} {language} {music_director or ''}",
            f"{movie_name} {year} {language} {hero or ''}",
            f"{movie_name} {language}",
            f"{movie_name} {year}",
            movie_name
        ]

        for query in search_phrases:
            print(f"🔍 Spotify Search Query: {query}")
            logging.info(f"Spotify Search: {query}")
            results = sp.search(q=query.strip(), type="album", limit=20, market=market)
            albums = results.get("albums", {}).get("items", [])

            if not albums:
                continue

            best_album = None
            best_score = -1
            best_similarity = 0.0

            for album in albums:
                score, sim = match_album_heuristic(album, movie_name, year, language, music_director, hero)
                if score > best_score or (score == best_score and sim > best_similarity):
                    best_score = score
                    best_similarity = sim
                    best_album = album

            if best_album and best_score >= 2:
                print(f"✅ Matched: {best_album['name']} by {[a['name'] for a in best_album['artists']]} (score={best_score}, sim={best_similarity:.2f})")
                logging.info(f"Selected album: {best_album['name']} (score={best_score})")
                return best_album["id"]

            elif best_album:
                print(f"⚠️ Weak fallback: {best_album['name']} (score={best_score}, sim={best_similarity:.2f}) — skipping.")
                logging.warning(f"Weak fallback album rejected: {best_album['name']}")

        logging.error(f"❌ No suitable album found for '{movie_name}' ({year}, {language})")
        return None

    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 429:
            logging.warning("⏳ Rate limit hit — switching Spotify client")
            switch_spotify_client()
            return search_album_on_spotify(movie_name, year, language, music_director, hero)
        logging.error(f"Spotify API error: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during album search: {e}")
        return None

    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 429:
            logging.warning("⏳ Rate limit hit. Switching Spotify client...")
            switch_spotify_client()
            return search_album_on_spotify(movie_name, year, language, music_director, hero)
        else:
            logging.error(f"Spotify API error: {e}")
            return None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
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

def fetch_lyrics_and_mood_from_gemini(song_name, artist_name):
    try:
        print("🔮 Using Gemini for lyrics and mood")
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        print(client)
        model_id = "gemini-2.0-flash"
        google_search_tool = Tool(google_search=GoogleSearch())

        prompt = f"""
Get the full lyrics and mood (from [happy, sad, energetic, calm]) of the song '{song_name}' by '{artist_name}'.
Respond in this format:

Mood: <one of happy, sad, energetic, calm>
Lyrics:
<lyrics here>
"""

        response = client.models.generate_content(
            model=model_id,
            contents=prompt.strip(),
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
            )
        )

        if response.candidates and response.candidates[0].content.parts:
            full_text = "\n".join(
                part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')
            )
            mood = None
            lyrics_lines = []
            for line in full_text.splitlines():
                if line.lower().startswith("mood:"):
                    mood = line.split(":", 1)[1].strip().lower()
                elif "lyrics" in line.lower():
                    continue
                else:
                    lyrics_lines.append(line)
            lyrics = "\n".join(lyrics_lines).strip()
            print(lyrics)
            print(mood)
            return lyrics, mood
        else:
            logging.warning("⚠ Gemini returned no content.")
            return "", ""

    except Exception as e:
        logging.warning(f"⚠ Gemini error for '{song_name}': {e}")
        return "", ""

def fetch_lyrics_and_mood(song_name, artist_name, album_name):
    try:
        song = genius.search_song(title=song_name, artist=artist_name)
        if song and song.lyrics:
            logging.info("📜 Got lyrics from Genius")
            save_lyrics(song.lyrics, album_name, song_name)
            return song.lyrics, None  # No mood from Genius
        else:
            logging.warning(f"⚠ Genius failed for '{song_name}', using Gemini...")
    except Exception as e:
        logging.warning(f"⚠ Genius exception for '{song_name}': {e}")

    lyrics, mood = fetch_lyrics_and_mood_from_gemini(song_name, artist_name)
    if lyrics:
        logging.info("📜 Got lyrics from Gemini")
        save_lyrics(lyrics, album_name, song_name)
    return lyrics, mood


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
            track_details = sp.tracks(track_ids[i:i + 50])["tracks"]

            for song_meta in track_details:
                song_name = song_meta["name"]
                artist_name = song_meta["artists"][0]["name"]
                logging.info(f"🎧 Processing: {song_name} by {artist_name}")

                audio_file = wav_file = None

                try:
                    audio_file = download_song_youtube(song_name, artist_name)
                    if not audio_file:
                        logging.warning(f"❌ Skipping download failure: {song_name}")
                        continue

                    wav_file = convert_mp3_to_wav(audio_file)
                    if not wav_file:
                        logging.warning(f"❌ Skipping conversion failure: {song_name}")
                        continue

                    audio_features = extract_audio_features(wav_file)
                    if not audio_features:
                        logging.warning(f"❌ Skipping feature extraction failure: {song_name}")
                        continue

                    lyrics, mood = fetch_lyrics_and_mood(song_name, artist_name, song_meta["album"]["name"])

                    song_record = {
                        "name": song_name,
                        "album": song_meta["album"]["name"],
                        "artist": ", ".join([artist["name"] for artist in song_meta["artists"]]),
                        "id": song_meta["id"],
                        "release_date": song_meta["album"]["release_date"],
                        "popularity": song_meta["popularity"],
                        "mood": mood or "unknown",
                        "lyrics_available": bool(lyrics),
                        **audio_features
                    }

                    songs_data.append(song_record)

                except Exception as song_err:
                    logging.error(f"❌ Failed to process '{song_name}': {song_err}")
                finally:
                    cleanup_files(filter(None, [audio_file, wav_file]))

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

                    # Harmonize columns for safe concat
                    all_columns = sorted(set(existing_df.columns) | set(output_df.columns))
                    existing_df = existing_df.reindex(columns=all_columns, fill_value="")
                    output_df = output_df.reindex(columns=all_columns, fill_value="")

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

    # Recursively find all CSVs
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
