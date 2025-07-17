import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import time
import random
import matplotlib.pyplot as plt

st.set_page_config(page_title="PawSound AI", layout="wide")

# Emotion categories
EMOTIONS = [
    {"name": "Happy", "emoji": "😊"},
    {"name": "Alert", "emoji": "⚠️"},
    {"name": "Anxious", "emoji": "😰"},
    {"name": "Calm", "emoji": "😌"},
    {"name": "Playful", "emoji": "⚡"},
]

st.title("🐕 PawSound AI - Dog Emotion Recognition")

uploaded_file = st.file_uploader("Upload a dog sound file (wav, mp3, flac)", type=["wav", "mp3", "flac"])

if uploaded_file:
    audio_data, sr = librosa.load(uploaded_file, sr=None)
    st.audio(uploaded_file)
    st.success(f"Audio loaded: {len(audio_data)/sr:.2f}s at {sr}Hz")

    # Show waveform using line chart
    st.subheader("📈 Audio Waveform")
    downsampled = audio_data[::int(sr/1000)]  # Downsample for performance
    st.line_chart(downsampled)

    # Simulate classification
    if st.button("🔍 Analyze Emotion"):
        with st.spinner("Analyzing audio..."):
            time.sleep(2)  # Simulate processing
            top_emotion = random.choice(EMOTIONS)
            confidence = round(random.uniform(60, 95), 1)

        st.subheader("🎯 Emotion Result")
        st.markdown(f"### {top_emotion['emoji']} **{top_emotion['name']}**")
        st.progress(confidence)
        st.write(f"Confidence Score: **{confidence}%**")

    with st.expander("ℹ️ What This Does"):
        st.markdown("""
        - Upload a dog's sound (bark, whimper, etc.)
        - The app extracts basic audio features
        - Simulates AI-based emotion recognition
        - Shows estimated emotion + confidence
        """)

else:
    st.info("Upload a file to get started.")
