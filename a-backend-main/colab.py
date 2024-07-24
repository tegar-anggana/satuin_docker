# @title revisi completeness percentage

import os
import ipywidgets as widgets
from IPython.display import display, Audio
from pydub import AudioSegment
from pydub.silence import split_on_silence
from huggingsound import SpeechRecognitionModel
from transformers import AutoTokenizer, AutoModel
import torch
import json
from sklearn.metrics.pairwise import cosine_similarity
from jiwer import wer, cer
import numpy as np

# Load the speech recognition model
model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-arabic")

# Load the tokenizer and model for Arabic RoBERTa
tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-ca")
roberta_model = AutoModel.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-ca")

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
        print(f"Surah {chosen_surah} not found in the database.")
        return

    surah_verses = [v['text'] for v in quran_embeddings[chosen_surah]]

    matched_verses = []
    total_similarity = 0
    total_wer = 0
    total_cer = 0
    correct_order = True
    incorrect_order_verses = []
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
        else:
            print(f"Segment {i + 1} has a high CER ({character_error_rate:.2f}), excluding from evaluation.")

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

    # Output results
    print(f"\nAverage Similarity: {avg_similarity * 100:.2f}%")
    print(f"Average WER: {avg_wer}")
    print(f"Average CER: {avg_cer}")
    print(f"Completeness Percentage: {completeness_percentage:.2f}%")

    if completeness_percentage < 100:
        print(f"The recitation is incomplete or has extra verses. Expected {len(surah_verses)} verses but got {total_segments} valid segments.")
        missing_verses = len(surah_verses) - total_segments
        print(f"Missing Verses: {missing_verses}")
    elif correct_order:
        print(f"The recitation of Surah {chosen_surah} is above the similarity threshold and passes.")
    else:
        print(f"The recitation of Surah {chosen_surah} does not pass due to incorrect verse order.")
        if incorrect_order_verses:
            print("Incorrectly ordered verses:")
            for verse, index in incorrect_order_verses:
                print(f"Transcribed Verse {index + 1}: '{verse}'")

    return completeness_percentage, avg_similarity

# Define ranges for silence detection parameters
silence_thresh_range = [-40, -35, -30, -25, -20]
min_silence_len_range = [300, 500, 700, 900]

# Load Quran embeddings database
with open("/content/quran_embeddings.json", "r") as f:
    quran_embeddings = json.load(f)

# Function to display and play audio
def display_audio(audio_path):
    display(Audio(audio_path, autoplay=True))

# Function to process audio for user input
def process_audio_for_user(audio_path, chosen_surah, silence_thresh_range, min_silence_len_range, threshold=0.95):
    best_similarity = 0
    best_params = (None, None)
    best_segments = None
    best_transcriptions = None

    # Perform grid search for best silence detection parameters
    for silence_thresh in silence_thresh_range:
        for min_silence_len in min_silence_len_range:
            print(f"Testing with silence_thresh={silence_thresh}, min_silence_len={min_silence_len}")

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
            correct_order = True
            for i, verse in enumerate(verses):
                if i >= len([v['text'] for v in quran_embeddings[chosen_surah]]):
                    break
                reference_text = quran_embeddings[chosen_surah][i]['text']
                reference_embedding = get_embeddings(reference_text).reshape(1, -1)
                input_embedding = get_embeddings(verse).reshape(1, -1)

                similarity = calculate_similarity(input_embedding, reference_embedding)
                total_similarity += similarity

                if similarity < threshold:
                    correct_order = False

            if verses:
                avg_similarity = total_similarity / len(verses)
            else:
                avg_similarity = 0

            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_params = (silence_thresh, min_silence_len)
                best_segments = segments
                best_transcriptions = transcriptions

    print(f"\nBest Parameters: silence_thresh={best_params[0]}, min_silence_len={best_params[1]}")
    print(f"Best Average Similarity: {best_similarity * 100:.2f}%")

    if best_segments and best_transcriptions:
        print("\nBest Segments and Transcriptions:")
        for i, segment in enumerate(best_segments):
            print(f"Segment {i + 1}: Duration={len(segment)} ms")

        print("\nTranscriptions:")
        for i, transcription in enumerate(best_transcriptions):
            print(f"Transcription {i + 1}: {transcription['transcription']}")

        completeness_percentage, avg_similarity = evaluate_transcription([t['transcription'] for t in best_transcriptions], quran_embeddings, chosen_surah, threshold)

        # Display final evaluation results
        print(f"\nFinal Evaluation:")
        print(f"Completeness Percentage: {completeness_percentage:.2f}%")
        print(f"Correctness: {avg_similarity * 100:.2f}%")

        if completeness_percentage == 100 and avg_similarity >= threshold:
            print("The recitation passes.")
        else:
            print("The recitation does not pass.")

# Main function for user input
def main():
    # Get surah choice from user
    chosen_surah = input("Enter the name of the surah: ")

    # Widget for file upload
    upload_widget = widgets.FileUpload(
        accept='.mp3,.wav',  # Accepted file types
        multiple=False  # Single file upload
    )
    display(upload_widget)

    def on_upload_change(change):
        # Check if a file was uploaded
        if upload_widget.value:
            # Get the uploaded file's info
            uploaded_file_info = list(upload_widget.value.values())[0]
            # Get the uploaded file's content
            uploaded_file_content = uploaded_file_info['content']
            # Write the uploaded content to a temporary file
            with open('uploaded_audio.wav', 'wb') as f:
                f.write(uploaded_file_content)

            # Display and play the uploaded audio
            display_audio('uploaded_audio.wav')

            # Process the audio file
            process_audio_for_user('uploaded_audio.wav', chosen_surah, silence_thresh_range, min_silence_len_range)

    upload_widget.observe(on_upload_change, names='value')

# Run the main function
main()