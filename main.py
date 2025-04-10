from flask import Flask, request, jsonify
import essentia.standard as ess
import os

app = Flask(__name__)

@app.route('/extract', methods=['POST'])
def extract_features():
    if 'file' not in request.files:
        return "Missing file", 400

    audio_file = 'downloaded_audio.wav'
    request.files['file'].save(audio_file)

    loader = ess.MonoLoader(filename=audio_file)
    audio = loader()

    features = {}
    features['tempo'] = ess.RhythmExtractor2013(method="multifeature")(audio)[0]
    features['loudness'] = ess.Loudness()(audio)
    features['pitch'] = ess.PitchYinFFT()(audio)[0]
    features['mfcc'] = list(ess.MFCC()(audio)[1][:13])  # Just first 13 MFCCs

    os.remove(audio_file)

    return jsonify(features)
