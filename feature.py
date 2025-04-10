import os
import librosa
import numpy as np
import pandas as pd
import subprocess

def download_youtube_audio(video_url, output_path="downloaded_audio.wav"):
    """Downloads YouTube audio using yt-dlp and converts it to WAV."""
    try:
        temp_file = "temp_audio.mp3"

        # Step 1: Download best audio format
        command = [
            "yt-dlp", "-f", "bestaudio", "--extract-audio", "--audio-format", "mp3", 
            "-o", temp_file, video_url
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Step 2: Convert MP3 to WAV
        output_wav = output_path
        subprocess.run(["ffmpeg", "-i", temp_file, "-ac", "1", "-ar", "22050", output_wav, "-y"], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        os.remove(temp_file)  # Cleanup temp file
        return output_wav

    except Exception as e:
        print(f"Failed to download or convert audio: {e}")
        return None


def extract_song_features(file_path):
    """Extracts features from an audio file using Librosa."""
    try:
        y, sr = librosa.load(file_path, sr=None)

        # Extract features
        tempo_array, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = tempo_array.item() if tempo_array.size > 0 else 0  # ✅ FIXED
        
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        energy = np.mean(librosa.feature.rms(y=y))
        loudness = np.mean(librosa.amplitude_to_db(np.abs(y), ref=np.max))  # ✅ FIXED
        speechiness = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

        # Fix acousticness & instrumentalness
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        acousticness = np.mean(spectral_rolloff) / sr  # ✅ Normalize (0-1)

        zero_crossings = librosa.feature.zero_crossing_rate(y=y)
        instrumentalness = 1 - np.mean(zero_crossings)  # ✅ Normalize (0-1)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        liveness = np.mean(spectral_bandwidth) / sr  # ✅ Normalize (0-1)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        valence = np.clip(np.mean(mfccs) / 100, 0, 1)  # ✅ Normalize (0-1)

        key = int(np.argmax(np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)))  
        time_signature = 4  
        duration_ms = round(librosa.get_duration(y=y, sr=sr) * 1000, 2)

        # Convert NumPy values to float before rounding
        features = {
            "Danceability": round(tempo / 200, 2),  
            "Acousticness": round(float(acousticness), 4),
            "Energy": round(float(energy), 4),
            "Instrumentalness": round(float(instrumentalness), 4),
            "Liveness": round(float(liveness), 4),
            "Valence": round(float(valence), 4),
            "Loudness": round(float(loudness), 4),
            "Speechiness": round(float(speechiness), 4),
            "Tempo": round(tempo, 2),
            "Key": key,
            "Time Signature": time_signature,
            "Duration (ms)": duration_ms
        }

        return features

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None


# ---- RUN SCRIPT ----
video_url = "https://youtu.be/KjatlEl2SuI?si=if_PrYHR0uj5fE90"
audio_file = download_youtube_audio(video_url)

if audio_file:
    features = extract_song_features(audio_file)
    if features:
        df = pd.DataFrame([features])  
        print(df)
