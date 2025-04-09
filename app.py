import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from datetime import datetime
from flask import Flask, jsonify, send_from_directory, render_template

# Initialize Flask app
app = Flask(__name__)

# Spotify API Credentials (Multiple Sets)
SPOTIFY_CREDENTIALS = [
    {"client_id": "15adf67aec934fe792bee0d467742326", "client_secret": "d03b2411aad24b8e80f3257660f9f10f"},
    {"client_id": "241765db513d43218e1e996b7d13d73f", "client_secret": "0fb1d0f0eed44f2e98d0e022335dd9e1"},
    {"client_id": "56bfb61f27234852826fd13e813174e6", "client_secret": "401f40941cba4f5bb2a0274f9fb34df2"}
]

# Track the active credential index
credential_index = 0

def get_spotify_client():
    """Initialize and return a Spotify client using the current credentials."""
    global credential_index
    creds = SPOTIFY_CREDENTIALS[credential_index]
    client_credentials_manager = SpotifyClientCredentials(
        client_id=creds["client_id"],
        client_secret=creds["client_secret"]
    )
    return spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Initialize the first Spotify client
sp = get_spotify_client()

# Input and Output Paths
MOVIE_CSV_FOLDER = "movie_csvs"
OUTPUT_FOLDER = "output_songs"
PROGRESS_FILE = "processed_movies.txt"  # Track processed movies

# Ensure output base folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def switch_spotify_client():
    """Switch to the next set of Spotify API credentials when rate-limited."""
    global credential_index, sp
    credential_index = (credential_index + 1) % len(SPOTIFY_CREDENTIALS)
    sp = get_spotify_client()
    print(f"Switched to Spotify credentials set {credential_index + 1}")

def search_album_on_spotify(movie_name, year, language):
    """Search for the movie album on Spotify using movie name, release year, and language."""
    try:
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
            print("Rate limit reached! Switching credentials...")
            switch_spotify_client()
            return search_album_on_spotify(movie_name, year, language)
        else:
            print(f"Spotify API error for {movie_name} ({year}, {language}): {e}")

    return None

def get_songs_from_album(album_id, movie_name):
    """Fetch song metadata from Spotify album ID."""
    songs_data = []
    if album_id:
        try:
            tracks = sp.album_tracks(album_id)
            for track in tracks["items"]:
                song_meta = sp.track(track["id"])
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
            print(f"Error fetching songs from album {album_id}: {e}")
    return songs_data

def save_progress(processed_movies):
    """Save processed movie names to a progress file."""
    with open(PROGRESS_FILE, "w") as f:
        for movie in processed_movies:
            f.write(movie + "\n")

def load_progress():
    """Load processed movie names from the progress file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return set(line.strip() for line in f)
    return set()

def process_movie_csv(file_path, processed_movies):
    """Process each movie CSV file and collect song details."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    for _, row in df.iterrows():
        movie_name = row["Title"]
        if movie_name in processed_movies:
            print(f"Skipping {movie_name} (already processed)")
            continue  # Skip already processed movies

        try:
            year = datetime.strptime(str(row["Release Date"]), "%d-%m-%Y").year
        except ValueError:
            print(f"Invalid date format for movie {movie_name}")
            continue
        
        language = row["Language"]

        year_folder = os.path.join(OUTPUT_FOLDER, str(year))
        language_folder = os.path.join(year_folder, language)
        os.makedirs(language_folder, exist_ok=True)

        output_file = os.path.join(language_folder, f"{year}_{language}.csv")

        album_id = search_album_on_spotify(movie_name, year, language)

        if album_id:
            songs = get_songs_from_album(album_id, movie_name)

            if songs:
                df_songs = pd.DataFrame(songs)
                if os.path.exists(output_file):
                    df_songs.to_csv(output_file, mode='a', index=False, header=False)
                else:
                    df_songs.to_csv(output_file, index=False)
                print(f"Updated: {output_file}")
            else:
                print(f"No songs found for {movie_name}")
        else:
            print(f"No album found for: {movie_name}")

        processed_movies.add(movie_name)
        save_progress(processed_movies)

@app.route("/")
def dashboard():
    """Render the dashboard UI."""
    return render_template("index.html")

@app.route("/progress")
def get_progress():
    """Return the list of processed movies."""
    processed_movies = load_progress()
    return jsonify({"movies": list(processed_movies)})

@app.route("/files")
def get_files():
    """List all generated CSV files."""
    file_list = []
    for root, _, files in os.walk(OUTPUT_FOLDER):
        for file in files:
            if file.endswith(".csv"):
                file_list.append(os.path.relpath(os.path.join(root, file), OUTPUT_FOLDER))
    return jsonify({"files": file_list})

@app.route("/download/<path:filename>")
def download_file(filename):
    """Allow downloading of processed CSV files."""
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    # Load progress before starting
    processed_movies = load_progress()

    # Process all CSV files
    for csv_file in os.listdir(MOVIE_CSV_FOLDER):
        if csv_file.endswith(".csv"):
            process_movie_csv(os.path.join(MOVIE_CSV_FOLDER, csv_file), processed_movies)

    print("Data collection completed!")
    app.run(debug=True)
