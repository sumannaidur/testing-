import requests
import pandas as pd
import time
import os
import logging
from flask import Flask, send_from_directory

# API Key
API_KEY = '51a322139dc6ff44903e5da693008149'  # Replace with your TMDb API key

# Flask App
app = Flask(__name__)

# Constants
BASE_URL = 'https://api.themoviedb.org/3/discover/movie'
CREDITS_URL = 'https://api.themoviedb.org/3/movie/{movie_id}/credits'
YEARS = list(range(2000, 2027))
LANGUAGES = {'hi': 'Hindi', 'kn': 'Kannada', 'ta': 'Tamil', 'te': 'Telugu'}
OUTPUT_FOLDER = 'movies_by_language'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_cast(movie_id):
    """Fetch cast details and return hero & heroine names."""
    try:
        response = requests.get(CREDITS_URL.format(movie_id=movie_id), params={'api_key': API_KEY})
        response.raise_for_status()
        data = response.json()

        hero, heroine = None, None
        for actor in data.get('cast', []):
            if actor.get('gender') == 2 and not hero:  # Gender 2 = Male (Hero)
                hero = actor.get('name')
            elif actor.get('gender') == 1 and not heroine:  # Gender 1 = Female (Heroine)
                heroine = actor.get('name')
            if hero and heroine:
                break  # Stop once both are found

        return hero, heroine
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching cast for movie ID {movie_id}: {e}")
        return None, None

def fetch_movies(lang_code, lang_name):
    """Fetch movies for a given language and save as CSV."""
    movie_data = []
    logging.info(f"Fetching movies for: {lang_name}")

    for year in YEARS:
        page = 1
        logging.info(f"  Year {year}...")

        while True:
            params = {
                'api_key': API_KEY,
                'language': 'en-US',
                'sort_by': 'popularity.desc',
                'include_adult': False,
                'include_video': False,
                'page': page,
                'primary_release_year': year,
                'with_original_language': lang_code
            }

            try:
                response = requests.get(BASE_URL, params=params)
                if response.status_code == 429:
                    logging.warning("Rate limit exceeded! Waiting 10 seconds before retrying...")
                    time.sleep(10)
                    continue  # Retry the request

                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching data: {e}")
                break

            if 'results' not in data or not data['results']:
                break

            for movie in data['results']:
                movie_id = movie.get('id')
                hero, heroine = get_cast(movie_id)

                movie_data.append({
                    'Title': movie.get('title'),
                    'Release Date': movie.get('release_date'),
                    'Language': lang_name,
                    'Hero': hero,
                    'Heroine': heroine,
                    'Overview': movie.get('overview'),
                    'Vote Average': movie.get('vote_average'),
                    'Popularity': movie.get('popularity'),
                })

            page += 1
            if page > data.get('total_pages', 1):
                break

            time.sleep(0.5)  # Be polite to the API

    # Save to CSV
    df = pd.DataFrame(movie_data)
    filepath = os.path.join(OUTPUT_FOLDER, f"{lang_name.lower()}_movies.csv")
    df.to_csv(filepath, index=False)
    logging.info(f"✅ Saved {len(movie_data)} movies to '{filepath}'")

# Run the script
for code, name in LANGUAGES.items():
    fetch_movies(code, name)

logging.info("\n🎉 All movie details from 2000–2026 saved successfully!")  

# Flask Route for Download
@app.route("/")
def home():
    return "App is running!"

@app.route('/download/<language>')
def download_file(language):
    filename = f"{language.lower()}_movies.csv"
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Render requires a web service running
