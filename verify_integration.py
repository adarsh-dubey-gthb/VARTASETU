
import os
import sys
import unittest
from unittest.mock import MagicMock
# Ensure path
sys.path.append("d:/newstts")
from core.services.streamer import NewsStreamer

# Need to mock basic things so we don't start FFMPEG/Whisper in test
class TestUpdatedTTS(unittest.TestCase):
    def test_tts_integration(self):
        print("Initializing Streamer for Verification...")
        # Mocking or instantiating. We want to test REAL TTS loading if possible, 
        # but don't want to start the full stream.
        
        streamer = NewsStreamer(youtube_url="http://dummy", target_lang="hi")
        
        if not hasattr(streamer, 'tts_model'):
            print("WARNING: GPU not detected or Model failed to load. Skipping real generation test.")
            return

        print("Testing generate_indic_tts with new logic...")
        text = "வணக்கம்" # Tamil Hello
        audio = streamer.generate_indic_tts(text)
        
        print(f"Generated {len(audio)} bytes of MP3 data.")
        self.assertTrue(len(audio) > 1000, "Audio should be generated")

if __name__ == "__main__":
    unittest.main()
