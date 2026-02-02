<p align="center">
  <img src="https://img.icons8.com/fluency/96/000000/translation.png" alt="VARTA-SETU Logo" width="100" height="100">
</p>

<h1 align="center">VARTA-SETU</h1>

<p align="center">
  <strong>Bridging the Language Divide: Real-Time Global News Translation into Indian Vernaculars</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Build-Passing-brightgreen" alt="Build Status">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-orange" alt="License">
  <img src="https://img.shields.io/badge/Hackathon-AI--FOR--BHARAT-red" alt="Hackathon">
  <img src="https://img.shields.io/badge/AWS-Polly%20%26%20Bedrock-yellow" alt="AWS Services">
</p>

---

## 📑 Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Tech Stack](#tech-stack)
4. [High-Level Architecture](#high-level-architecture)
5. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Configuration](#configuration)
6. [Usage](#usage)
7. [System Scripts Overview](#system-scripts-overview)
8. [Folder Structure](#folder-structure)
9. [Roadmap](#roadmap)
10. [Contributing](#contributing)
11. [License](#license)
12. [Contact & Acknowledgements](#contact--acknowledgements)

---

## 🚀 Introduction

**VARTA-SETU** (meaning "News Bridge") is an advanced AI-driven pipeline developed for the **AI-FOR-BHARAT Hackathon**. In an era where global news shapes local perspectives, language remains a formidable barrier for millions across the Indian subcontinent. While major international news outlets like CNN, BBC, and Al Jazeera provide world-class reporting, their content is often inaccessible to the non-English speaking population of India.

VARTA-SETU solves this by capturing LIVE international news streams from YouTube and performing real-time speech-to-speech translation into various Indian languages (Hindi, Marathi, Tamil, Telugu, etc.). By integrating state-of-the-art Speech-to-Text (STT), Neural Machine Translation (NMT), and Text-to-Speech (TTS) technologies, the project creates a seamless, low-latency bridge between global events and local understanding. 

Whether it's a farmer in Vidarbha wanting to understand global market shifts or a student in Bihar following international space missions, VARTA-SETU democratizes information by delivering it in the language of their heart.

---

## ✨ Key Features

- 🎙️ **Live Stream Processing**: Utilizes `yt-dlp` to hook into live YouTube broadcasts, extracting audio buffers on-the-fly without the need for massive local storage.
- ⚡ **High-Performance Transcription**: Employs `Faster-Whisper` (a specialized implementation of OpenAI's Whisper) to convert spoken English into text with industry-leading accuracy and speed.
- 🌍 **Context-Aware Translation**: Beyond simple word-to-word replacement, the system uses `deep-translator` and `Amazon Bedrock (Claude 3.5 Sonnet)` to ensure cultural nuances and technical terms are translated accurately into Indian contexts.
- 🗣️ **Dual-Engine TTS**:
    - **Amazon Polly**: High-fidelity, neural-quality voices for a professional broadcast feel.
    - **Indic TTS**: Specialized models tailored for the unique phonetic requirements of Indian languages to ensure natural prosody and intonation.
- 🔄 **Asynchronous Pipeline**: Built on **Django Channels** and **Daphne**, the architecture handles long-running AI tasks without blocking the main user interface, providing a smooth streaming experience.
- 🛠️ **Error-Resilient Design**: Includes verification scripts for every stage of the pipeline (model loading, audio output, integration checks).

---

## 🛠️ Tech Stack

### **Core Frameworks & Languages**
- **Python 3.10+**: The backbone of the entire data processing and AI pipeline.
- **Django**: Used as the robust web backend for managing user sessions and system state.
- **Django Channels/Daphne**: For handling WebSockets and asynchronous communication required for live audio streams.

### **AI & Machine Learning**
- **Faster-Whisper**: For lightning-fast Speech-to-Text conversion.
- **Parler-TTS**: For advanced, high-quality speech synthesis options.
- **Amazon Bedrock (Claude 3.5 Sonnet)**: Orchestrating complex linguistic refinements and ensuring the "soul" of the news is preserved during translation.
- **PyTorch**: The underlying deep learning framework for model execution.

### **Cloud & Services**
- **Amazon Polly**: Neural TTS for generating lifelike Indian-language voices.
- **Boto3**: The AWS SDK for Python, used for seamless integration with Polly and Bedrock.

### **Audio/Video Processing**
- **yt-dlp**: For extracting audio streams from live YouTube sources.
- **Pydub & Soundfile**: For manipulating, slicing, and refining audio segments.
- **FFmpeg**: The heavy lifter for media transcoding and stream handling.

---

## 🏗️ High-Level Architecture

```mermaid
graph LR
    A[Live YouTube Stream] --> B(yt-dlp)
    B --> C{Audio Processor}
    C --> D[Faster-Whisper STT]
    D --> E[Deep-Translator / Bedrock]
    E --> F{TTS Selection}
    F --> G[Amazon Polly]
    F --> H[Indic TTS Models]
    G --> I[Final Audio Sync]
    H --> I
    I --> J[End User/Web Dashboard]
```

1. **Ingestion**: The system connects to a live URL and extracts audio chunks.
2. **Analysis**: The STT engine converts the English audio to raw text.
3. **Synthesis**: The text is translated into the target Indian language.
4. **Generation**: The translated text is sent to the TTS engine (Polly or Indic TTS).
5. **Delivery**: The synthesized audio is streamed back to the user or saved for broadcast.

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.10 or higher installed.
- **FFmpeg** installed on your system path (crucial for `pydub` and `yt-dlp`).
- An active **AWS Account** with permissions for Polly and Bedrock (Claude 3.5 access).
- (Optional) NVIDIA GPU with CUDA drivers for faster local inference of Whisper models.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/varta-setu.git
   cd varta-setu
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # Install the specialized parler-tts from source
   pip install git+https://github.com/huggingface/parler-tts.git
   ```

### Configuration

Create a `.env` file in the root directory based on `.env.example`:

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
INDIC_TTS_CONFIG_PATH=./config/indic_tts.json
DEBUG=True
```

---

## 🚀 Usage

### 1. Running the Web Server
To launch the Django-based dashboard:
```bash
python manage.py migrate
python manage.py runserver
```

### 2. Testing Component Integrations
Before running the full pipeline, verify your setups using the built-in diagnostic tools:
- **Test YouTube Extraction**: `python test_ytdlp.py`
- **Test AWS Polly**: `python test_polly.py`
- **Test Indic TTS**: `python test_indic_tts.py`
- **Check Model Load**: `python debug_model_load.py`

### 3. Running the Translation Pipeline
To start the live translation of a specific YouTube stream:
```bash
python refined_indic_tts.py --url https://www.youtube.com/watch?v=live_stream_id --lang hi
```
*(Replace `hi` with your target language code: `mr`, `ta`, `te`, etc.)*

---

## 🛠️ System Scripts Overview

The repository contains several specialized scripts to ensure the pipeline remains healthy:
- `verify_pipeline.py`: A comprehensive end-to-end check from audio input to translated audio output.
- `check_indictts_config.py`: Validates the local configuration for Indian language models.
- `test_audio_debug.py`: Used to troubleshoot soundcard or audio driver issues during output.
- `manage.py`: Standard Django management script for database and server tasks.

---

## 📂 Folder Structure

```text
VARTA-SETU/
├── config/                 # Configuration files for TTS and AWS
├── core/                   # Core business logic for translation
├── debug_model_load.py     # Diagnostic tool for local ML models
├── manage.py               # Django entry point
├── refined_indic_tts.py    # Main pipeline logic
├── requirements.txt        # Python dependencies
├── result.txt              # Logging/Output text storage
├── static/                 # CSS/JS for the dashboard
├── templates/              # HTML for the web UI
├── tests/                  # Integration and unit tests
└── .env.example            # Environment variable template
```

---

## 🗺️ Roadmap

- [ ] **Low-Latency Streaming**: Reduce the gap between original broadcast and translated audio to < 5 seconds.
- [ ] **Multi-Speaker Identification**: Distinguish between different news anchors and assign different voices.
- [ ] **Mobile Application**: Flutter-based app for users to listen to news on the go.
- [ ] **Sentiment-Aware TTS**: Adjusting the tone of the synthesized voice based on the news category (e.g., somber for tragic news, energetic for sports).
- [ ] **Expanded Dialect Support**: Moving beyond standard Hindi to regional dialects like Bhojpuri or Marwari.

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 📞 Contact & Acknowledgements

**Project Lead:** [Your Name/Team Name]  
**Hackathon:** [AI-FOR-BHARAT](https://aiforbharat.iitm.ac.in/)

- **Special Thanks**: 
  - The **Bhashini** team for their inspiring work in Indian language datasets.
  - **AWS** for providing credits and robust infrastructure for Polly and Bedrock.
  - The creators of **Faster-Whisper** for making high-speed STT accessible.

<p align="center">
  Built with ❤️ for a more connected India.
</p>
