import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from core.services.streamer import NewsStreamer

def test_polly():
    print("Testing Amazon Polly Integration...")
    
    # Check env vars (Masked)
    if os.environ.get('AWS_ACCESS_KEY_ID'):
        print("AWS_ACCESS_KEY_ID is set.")
    else:
        print("WARNING: AWS_ACCESS_KEY_ID is NOT set. Polly might fail.")

    try:
        # Force Hindi to test Aditi fallback
        streamer = NewsStreamer("http://dummy-url.com", target_lang='hi')
        
        if streamer.polly_client:
            print("Polly Client initialized successfully.")
        else:
            print("Polly Client FAILED to initialize.")
            return

        text = "यह अमेज़न पोली एकीकरण का एक परीक्षण है।" # Hindi test text
        print(f"Generating TTS for: '{text}'")
        
        audio_bytes = streamer.generate_tts(text)
        
        with open("result.txt", "w") as f:
            if audio_bytes and len(audio_bytes) > 0:
                print(f"SUCCESS: Generated {len(audio_bytes)} bytes of audio.")
                f.write(f"SUCCESS: {len(audio_bytes)}")
            else:
                print("FAILURE: Generated 0 bytes.")
                f.write("FAILURE: 0 bytes")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_polly()
