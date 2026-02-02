import logging
import time
import sys
import os
import queue

# Ensure d:\newstts is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.services.streamer import NewsStreamer

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_pipeline():
    # Use a known safe YouTube video (not live stream if possible, or a stable live stream)
    # This is a short video: "Test Video"
    # TEST_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw" 
    # Use ACTUAL Live Stream URL from views.py
    TEST_URL = "https://www.youtube.com/live/6ygySTWJ92M?si=B5wOByb_gP4aLnGG"
    
    print(f"Initializing NewsStreamer with URL: {TEST_URL}")
    streamer = NewsStreamer(youtube_url=TEST_URL, target_lang="hi")
    
    print("Starting streaming...")
    # streamer.start_streaming() # audio_generator calls this internally
    
    start_time = time.time()
    max_duration = 60 # Run for 60 seconds max
    
    try:
        gen = streamer.audio_generator()
        for i, audio_chunk in enumerate(gen):
            if time.time() - start_time > max_duration:
                print("TIMEOUT")
                break
                
            print(f"SUCCESS: Received Chunk #{i} of size {len(audio_chunk)} bytes! Time: {time.time() - start_time:.2f}s")
            
            # Skip the first chunk (Immediate Silence)
            if i == 0:
                print("DEBUG: Received first chunk (Silence Start). Skipping...")
                continue
            
            # If chunk is small (keep-alive silence), log and continue
            if len(audio_chunk) < 5000:
                print("DEBUG: Received small chunk (keep-alive). Waiting for real audio...")
                continue
            
            # Save first real chunk and exit
            with open("pipeline_output.mp3", "wb") as f:
                f.write(audio_chunk)
            print("Saved pipeline_output.mp3. Exiting test.")
            return

    except Exception as e:
        print(f"Error in generator loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        streamer.stop()

if __name__ == "__main__":
    verify_pipeline()
