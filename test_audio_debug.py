import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import io
import numpy as np
from pydub import AudioSegment

def test_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    
    sample_rate = model.config.sampling_rate
    print(f"Model Sample Rate: {sample_rate} Hz")
    
    text = "Namaste. This is a test to check the audio quality. Does this sound like a human or an alien? I hope it is clear."
    prompt = "A female speaker delivers a very clear, high-quality, and expressive speech with a moderate speed and pitch."
    
    print("Generating audio...")
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    
    print(f"Raw Audio Max Amplitude: {np.abs(audio_arr).max()}")
    
    # 1. Save Raw WAV
    print("Saving demo_raw.wav...")
    sf.write("demo_raw.wav", audio_arr, sample_rate)
    
    # 2. Normalize
    max_val = np.abs(audio_arr).max()
    if max_val > 1.0:
        audio_arr_norm = audio_arr / max_val
    else:
        audio_arr_norm = audio_arr
        
    print("Saving demo_normalized.wav...")
    sf.write("demo_normalized.wav", audio_arr_norm, sample_rate)
    
    # 3. Convert to MP3 (Simulating Streamer)
    print("Saving demo_mp3_192k.mp3...")
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio_arr_norm, sample_rate, format='WAV')
    wav_buffer.seek(0)
    
    audio_segment = AudioSegment.from_wav(wav_buffer)
    audio_segment.export("demo_mp3_192k.mp3", format="mp3", bitrate="192k")
    
    print("Done! Please check the output files.")

if __name__ == "__main__":
    test_generation()
