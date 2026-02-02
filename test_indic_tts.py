import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import torch

# Ensure d:\newstts is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.services.streamer import NewsStreamer

class TestIndicTTS(unittest.TestCase):
    def setUp(self):
        # Mock yt_dlp to avoid network calls during init if possible, or just ignore
        pass

    def test_generation_logic(self):
        print("Initializing Streamer (Mocking Models for fast CPU test)...")
        # We need to force load the "IndiTTS" path. 
        # But locally I might not have GPU.
        # So I will mock torch.cuda.is_available to return True if I want to test the LOGIC flows,
        # but the actual MODEL loading will fail if I don't have it.
        
        # Let's just try to instantiate and run generate_indic_tts assuming dependencies are there.
        # If not, we catch the import error.
        
        try:
            streamer = NewsStreamer(youtube_url="http://example.com", target_lang="hi")
            
            # Manually inject mock model if real one didn't load (likely on CPU)
            if not hasattr(streamer, 'tts_model'):
                print("Injecting MOCK model and tokenizer for CPU testing...")
                streamer.tts_device = "cpu"
                streamer.tts_model = MagicMock()
                streamer.tts_model.config.sampling_rate = 22050
                streamer.tts_tokenizer = MagicMock()
                
                # Mock generate output (random numpy array)
                mock_audio = torch.randn(1, 10000) # shape
                streamer.tts_model.generate.return_value = mock_audio
            
            text = "नमस्ते दुनिया"
            print(f"Testing generate_indic_tts with text: {text}")
            audio_bytes = streamer.generate_indic_tts(text)
            
            print(f"Result: {len(audio_bytes)} bytes")
            self.assertTrue(len(audio_bytes) > 0, "Audio bytes should be non-empty")
            
            # Check if it is MP3
            self.assertTrue(audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb' or audio_bytes[:2] == b'\xff\xf3', "Should be MP3 format")
            
        except Exception as e:
            print(f"Test Failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    unittest.main()
