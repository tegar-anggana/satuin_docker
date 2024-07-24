from flask import Flask, request, jsonify, send_file
from pydub import AudioSegment
from pydub.silence import split_on_silence
from huggingsound import SpeechRecognitionModel
from transformers import AutoTokenizer, AutoModel
import tempfile
import torch
import json
from sklearn.metrics.pairwise import cosine_similarity
from jiwer import wer, cer
import numpy as np
import os

app = Flask(__name__)

# Load the speech recognition model
model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-arabic")

# Load the tokenizer and model for Arabic RoBERTa
tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-ca")
roberta_model = AutoModel.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-ca")

# Load Quran embeddings database
with open("new_quran_embeddings.json", "r", encoding="utf8") as f:
    quran_embeddings = json.load(f)

# Define ranges for silence detection parameters
silence_thresh_range = [-40, -35, -30, -25, -20]
min_silence_len_range = [300, 500, 700, 900]

# Function to split audio into segments based on silence and resample to 16000 Hz
def split_audio_on_silence(audio_path, silence_thresh=-40, min_silence_len=500, target_sample_rate=16000):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(target_sample_rate)  # Resample to target sample rate
    segments = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    return segments

# Function to transcribe a list of audio segments
def transcribe_segments(segments, target_sample_rate=16000):
    transcriptions = []
    for i, segment in enumerate(segments):
        segment = segment.set_frame_rate(target_sample_rate)  # Ensure segment is resampled
        segment.export(f"segment_{i}.wav", format="wav")
        transcriptions.extend(model.transcribe([f"segment_{i}.wav"]))
    return transcriptions

# Function to get embeddings from text using RoBERTa
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Function to normalize embeddings
def normalize_embeddings(embeddings):
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norm
    return normalized_embeddings

# Function to calculate cosine similarity
def calculate_similarity(input_embedding, reference_embedding):
    input_embedding = normalize_embeddings(input_embedding)
    reference_embedding = normalize_embeddings(reference_embedding)
    similarity = cosine_similarity(input_embedding, reference_embedding)[0][0]
    similarity = max(0, min(1, similarity))  # Ensuring similarity stays in [0, 1]
    return similarity

# Function to calculate WER and CER
def calculate_wer_cer(reference, hypothesis):
    word_error_rate = wer(reference, hypothesis)
    character_error_rate = cer(reference, hypothesis)
    return word_error_rate, character_error_rate

# Function to evaluate the transcribed verses against the Quran database
def evaluate_transcription(verses, quran_embeddings, chosen_surah, threshold=0.95):
    if chosen_surah not in quran_embeddings:
        return {"error": f"Surah {chosen_surah} not found in the database."}, 400

    surah_verses = [v['text'] for v in quran_embeddings[chosen_surah]]

    matched_verses = []
    total_similarity = 0
    total_wer = 0
    total_cer = 0
    total_segments = 0  # Counter for total valid segments

    for i, verse in enumerate(verses):
        if i >= len(surah_verses):
            break

        reference_text = surah_verses[i]
        reference_embedding = get_embeddings(reference_text).reshape(1, -1)
        input_embedding = get_embeddings(verse).reshape(1, -1)

        similarity = calculate_similarity(input_embedding, reference_embedding)

        # Calculate WER and CER
        word_error_rate, character_error_rate = calculate_wer_cer(reference_text, verse)

        # Only count segments with CER <= 0.7 towards completeness percentage
        if character_error_rate <= 0.7:
            total_segments += 1
            total_similarity += similarity
            total_wer += word_error_rate
            total_cer += character_error_rate
            matched_verses.append(similarity)

    # Calculate average similarity and error rates
    if total_segments > 0:
        avg_similarity = total_similarity / total_segments
        avg_wer = total_wer / total_segments
        avg_cer = total_cer / total_segments
    else:
        avg_similarity = 0
        avg_wer = 0
        avg_cer = 0

    # Calculate completeness percentage
    completeness_percentage = (total_segments / len(surah_verses)) * 100

    return {
        "avg_similarity": avg_similarity * 100,
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "completeness_percentage": completeness_percentage
    }, 200

# Function to process audio for user input
def process_audio_for_user(audio_path, chosen_surah, silence_thresh_range, min_silence_len_range, threshold=0.95):
    best_similarity = 0
    best_params = (None, None)
    best_segments = None
    best_transcriptions = None

    # Perform grid search for best silence detection parameters
    for silence_thresh in silence_thresh_range:
        for min_silence_len in min_silence_len_range:
            # Split audio into segments based on current parameters
            segments = split_audio_on_silence(audio_path, silence_thresh=silence_thresh, min_silence_len=min_silence_len)

            if not segments:
                continue

            # Transcribe each segment
            transcriptions = transcribe_segments(segments)

            if not transcriptions:
                continue

            # Collect all transcribed verses
            verses = [t['transcription'] for t in transcriptions]

            # Evaluate transcriptions
            total_similarity = 0
            for i, verse in enumerate(verses):
                if i >= len([v['text'] for v in quran_embeddings[chosen_surah]]):
                    break
                reference_text = quran_embeddings[chosen_surah][i]['text']
                reference_embedding = get_embeddings(reference_text).reshape(1, -1)
                input_embedding = get_embeddings(verse).reshape(1, -1)

                similarity = calculate_similarity(input_embedding, reference_embedding)
                total_similarity += similarity

            if verses:
                avg_similarity = total_similarity / len(verses)
            else:
                avg_similarity = 0

            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_params = (silence_thresh, min_silence_len)
                best_segments = segments
                best_transcriptions = transcriptions

    if best_segments and best_transcriptions:
        completeness_percentage, avg_similarity = evaluate_transcription(
            [t['transcription'] for t in best_transcriptions],
            quran_embeddings,
            chosen_surah,
            threshold
        )
        # return {
        #     "best_params": {
        #         "silence_thresh": best_params[0],
        #         "min_silence_len": best_params[1]
        #     },
        #     "completeness_percentage": completeness_percentage,
        #     "avg_similarity": avg_similarity * 100
        # }

        if completeness_percentage["completeness_percentage"] == 100:
            status_kelulusan = 'LULUS'
        else:
            print("The recitation does not pass.")
            status_kelulusan = 'TIDAK LULUS'
        
        completeness_percentage_round = f"{completeness_percentage['completeness_percentage']:.2f}%"
        
        return {
            "persentase_kelengkapan": completeness_percentage_round,
            "status_kelulusan": status_kelulusan
        }
    else:
        return {"error": "No valid segments or transcriptions found."}, 400

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files or 'surah' not in request.form:
        return jsonify({"error": "Missing file or surah in request"}), 400

    file = request.files['file']
    surah = request.form['surah']
    audio_path = 'uploaded_audio.wav'

    # Check if the file is in MP3 format and convert it to WAV if needed
    if file.filename.endswith('.mp3'):
        audio = AudioSegment.from_file(file, format='mp3')
        audio.export(audio_path, format='wav')
    elif file.filename.endswith('.wav'):
        file.save(audio_path)
    else:
        return jsonify({"error": "No selected file"}), 400

    result = process_audio_for_user(audio_path, surah, silence_thresh_range=[-40, -35, -30, -25, -20], min_silence_len_range=[300, 500, 700, 900])
    os.remove(audio_path)  # Clean up the temporary file
    return jsonify(result)

    return jsonify({"error": "Invalid file type"}), 400

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'wav', 'mp3'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/surahs', methods=['GET'])
def get_surahs():
    surahs = list(quran_embeddings.keys())
    return jsonify(surahs)

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Make sure to replace '0.0.0.0' with your actual local IP if necessary