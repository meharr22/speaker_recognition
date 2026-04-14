# 🎙️ VoicePrint – Speaker Recognition System

A modern web-based speaker recognition application that allows users to **create personalized voice datasets, train machine learning models, and identify speakers from audio inputs** in real time.

Built with an intuitive UI using Streamlit, this project leverages **MFCC-based feature extraction** and a **Support Vector Machine (SVM)** classifier to deliver fast and reliable predictions.

---



## ✨ Key Highlights

- 🎤 Record or upload audio to build your own speaker dataset  
- 🧠 Train a machine learning model directly from the browser  
- 🔍 Predict speaker identity from new audio samples  
- ⚡ Real-time inference using microphone input  
- 📉 Lightweight yet effective ML pipeline (MFCC + SVM)  
- 🎨 Simple, interactive, and user-friendly interface  

---

## 🛠️ Tech Stack

| Category        | Tools Used |
|----------------|-----------|
| Language        | Python |
| Frontend        | Streamlit |
| Audio Processing| Librosa |
| ML Framework    | Scikit-learn |
| Model Storage   | Joblib |
| Recording Tool  | streamlit-audiorecorder |

---
## 📁 Project Structure

```bash
speaker-recognition-app/
│── app.py                # Main Streamlit application

├── utils/
│   ├── feature.py        # MFCC feature extraction
│   ├── train.py          # Model training logic
│   └── predict.py        # Prediction pipeline

└── requirements.txt      # Project dependencies

```
## ⚙️ How It Works

### 1️⃣ Audio Processing
Raw audio signals are converted into **MFCC (Mel-Frequency Cepstral Coefficients)**, which capture the essential characteristics of human speech.

### 2️⃣ Feature Engineering
Each audio file is transformed into a structured numerical representation suitable for machine learning.

### 3️⃣ Model Training
An **SVM classifier** is trained on the extracted features to learn speaker-specific patterns.

### 4️⃣ Prediction
New audio input is processed and passed through the trained model to identify the speaker along with confidence scores.

---
## ⚙️ Installation & Setup

Follow these steps to run the project locally:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/meharr22/speaker_recognition.git
cd speaker-recognition-app
```

### 2️⃣ Create Virtual Environment (Recommended)
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application
```bash
streamlit run app.py
```

### 5️⃣ Open in Browser
Once the server starts, open:
```
http://localhost:8501
```
## 📌 Usage Guide

1. Enter the name of the speaker you want to register  
2. Upload audio files or record voice samples using the microphone  
3. Save the recordings to build your custom dataset  
4. Train the model directly from the interface  
5. Upload or record new audio to test speaker predictions  

---

## ⚠️ Limitations

- The system can only recognize speakers included in the training dataset  
- Performance may drop with noisy or low-quality audio recordings  
- The deployed version may not store data permanently  
- Limited scalability with traditional ML models like SVM  

---

## 🔮 Future Enhancements

- 💾 Persistent storage for datasets and trained models  
- 🤖 Integration of deep learning models (CNN, LSTM, or Transformers)  
- 🔊 Noise reduction and audio augmentation techniques  
- 👥 Multi-user support with authentication system  
- 📊 Advanced visualization of prediction confidence and analytics  

---

## 👨‍💻 Author

**Mehar Arora**
