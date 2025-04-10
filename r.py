import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# 🔑 Replace with your actual Spotify credentials
CLIENT_ID = "15adf67aec934fe792bee0d467742326"
CLIENT_SECRET = "d03b2411aad24b8e80f3257660f9f10f"

# Set up authentication
try:
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    print("✅ Spotify authentication successful!")
except Exception as e:
    print(f"❌ Spotify authentication failed: {e}")
    exit()

# Function to get song features
def get_song_features(song_name, artist_name):
    query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=query, type="track", limit=1)

    if results["tracks"]["items"]:
        track_id = results["tracks"]["items"][0]["id"]
        
        # Fetch audio features
        features = sp.audio_features([track_id])[0]

        if features:
            return {
                "Song": song_name,
                "Artist": artist_name,
                "Danceability": features["danceability"],
                "Energy": features["energy"],
                "Valence": features["valence"],
                "Tempo (BPM)": features["tempo"],
                "Loudness": features["loudness"],
                "Acousticness": features["acousticness"],
                "Instrumentalness": features["instrumentalness"],
                "Speechiness": features["speechiness"],
                "Liveness": features["liveness"]
            }
        else:
            return "❌ Failed to fetch audio features."
    else:
        return "❌ Song not found!"

# Test with a song
song_features = get_song_features("Blinding Lights", "The Weeknd")
print(song_features)
