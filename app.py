import streamlit as st
import os
from utils.train import train_model
from utils.predict import predict
from audiorecorder import audiorecorder
import os
os.makedirs("dataset", exist_ok=True)
os.makedirs("model", exist_ok=True)
DATASET_PATH = "dataset"
MODEL_PATH = "model/voice_model.pkl"

st.set_page_config(page_title="Speaker Recognition", layout="centered")

st.title("🎙️ Speaker Recognition System")

# ========================= TABS =========================
tab1, tab2 = st.tabs(["🧑‍🏫 Create Dataset", "🧪 Test Model"])

# ======================================================
# 🧑‍🏫 CREATE DATASET
# ======================================================
with tab1:
    st.header("Add Speaker Data")

    name = st.text_input("Enter Speaker Name")

    # ---------------- UPLOAD ----------------
    st.subheader("📁 Upload Audio")

    files = st.file_uploader(
        "Upload audio files",
        type=["wav", "flac"],
        accept_multiple_files=True
    )

    if st.button("Save Uploaded Data"):
        if name and files:
            folder = os.path.join(DATASET_PATH, name)
            os.makedirs(folder, exist_ok=True)

            for file in files:
                file_path = os.path.join(folder, file.name)

                if os.path.exists(file_path):
                    st.warning(f"⚠️ {file.name} already exists, skipping")
                    continue

                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

            st.success(f"✅ Uploaded data saved for {name}")
        else:
            st.error("❌ Provide name and files")

    st.divider()

    # ---------------- RECORD DATASET ----------------
    st.subheader("🎤 Record Audio for Dataset")

    record_audio = audiorecorder(
        "Click to record",
        "Recording...",
        key="dataset_recorder"
    )

    if len(record_audio) > 0:
        st.audio(record_audio.export().read())

        audio_name = st.text_input("Enter audio name (optional)")

        if st.button("Save Recorded Audio"):
            if name:
                folder = os.path.join(DATASET_PATH, name)
                os.makedirs(folder, exist_ok=True)

                if audio_name.strip() == "":
                    file_count = len(os.listdir(folder))
                    file_path = os.path.join(folder, f"record_{file_count}.wav")
                else:
                    file_path = os.path.join(folder, f"{audio_name}.wav")

                if os.path.exists(file_path):
                    st.error("❌ File already exists! Choose another name")
                else:
                    record_audio.export(file_path, format="wav")
                    st.success(f"✅ Saved as {os.path.basename(file_path)}")
            else:
                st.error("❌ Enter speaker name first")

    st.divider()

    # ---------------- TRAIN ----------------
    st.header("⚙️ Train Model")

    if st.button("Train Model"):
        labels = train_model(DATASET_PATH, MODEL_PATH)
        st.success("✅ Model trained successfully!")

# ======================================================
# 🧪 TEST MODEL
# ======================================================
with tab2:
    st.header("📁 Upload Audio")

    test_file = st.file_uploader("Upload test audio", type=["wav", "flac"])

    if test_file:
        st.audio(test_file)

        temp_path = "temp.wav"
        with open(temp_path, "wb") as f:
            f.write(test_file.getbuffer())

        labels = sorted(os.listdir(DATASET_PATH))

        if st.button("Predict Uploaded Audio"):
            speaker, conf = predict(temp_path, MODEL_PATH, labels)

            st.success(f"🎯 Speaker: {speaker}")
            st.write(f"📊 Confidence: {conf:.2f}")

    st.divider()

    # ---------------- RECORD TEST ----------------
    st.header("🎤 Record & Test")

    audio = audiorecorder(
        "Click to record",
        "Recording...",
        key="test_recorder"
    )

    if len(audio) > 0:
        st.audio(audio.export().read())

        temp_path = "recorded.wav"
        audio.export(temp_path, format="wav")

        labels = sorted(os.listdir(DATASET_PATH))

        if st.button("Predict Recorded Audio"):
            speaker, conf = predict(temp_path, MODEL_PATH, labels)

            st.success(f"🎯 Speaker: {speaker}")
            st.write(f"📊 Confidence: {conf:.2f}")
