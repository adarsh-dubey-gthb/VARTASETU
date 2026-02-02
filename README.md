# Varta Setu - Breaking Language Barriers in Live News 🌍

**Varta Setu** (Bridge of Conversation) is an AI-powered real-time news aggregation and translation platform designed to democratize access to global information. It enables users to consume live news broadcasts and articles in their native language, bridging the gap between global media and local understanding.

![Varta Setu Demo](https://via.placeholder.com/800x400?text=Varta+Setu+Dashboard)

## 🚀 Key Features

*   **Live Audio Translation**: Real-time streaming of news audio translated into Indian regional languages (Hindi, Tamil, Telugu, Bengali, etc.) using AI.
*   **AI-Powered News Generation**: Automatically generates concise summaries and headlines from live video feeds using **Gemini 1.5 Pro**.
*   **Multi-Language Interface**: A fully localized user interface that adapts to the selected language.
*   **Smart Categorization**: Categorizes news into Business, Sports, Tech, and more for easy navigation.
*   **Accessible Design**: High-contrast, easy-to-read UI with dark mode support.

## 🛠️ Tech Stack

*   **Backend**: Django (Python)
*   **AI Models**:
    *   **Content Generation**: Google Gemini 1.5 Pro
    *   **Transcription**: Faster-Whisper
    *   **Translation**: Google Translate (Deep-Translator)
    *   **Text-to-Speech**: Amazon Polly / Indic Parler TTS
*   **Streaming**: FFmpeg, YT-DLP
*   **Frontend**: HTML5, CSS3, JavaScript (Vanilla)

## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/adarsh-dubey-gthb/VARTASETU.git
    cd VARTASETU
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

    *Note: FFmpeg must be installed on your system and added to PATH.*

3.  **Environment Configuration**
    Create a `.env` file in the root directory:
    ```env
    GEMINI_API_KEY=your_gemini_key
    AWS_ACCESS_KEY_ID=your_aws_key
    AWS_SECRET_ACCESS_KEY=your_aws_secret
    DJANGO_DEBUG=True
    ```

4.  **Run Migrations**
    ```bash
    python manage.py migrate
    ```

5.  **Start Main Server**
    ```bash
    python manage.py runserver
    ```

## 🧠 AI Pipeline

1.  **Ingest**: Fetches live YouTube audio streams.
2.  **Transcribe**: Converts audio to text using `Faster-Whisper`.
3.  **Translate**: Translates text to target language (e.g., Hindi) with context awareness.
4.  **Synthesize**: Generates natural-sounding speech using `Polly` or `Parler-TTS`.
5.  **Stream**: Delivers continuous audio stream to the client.

## 👥 Contributors

*   **Adarsh Dubey** - *Lead Developer*

---
*Built for Hackathon 2026. Empowering the next billion users.*
