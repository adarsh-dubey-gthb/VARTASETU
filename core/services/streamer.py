import os
import time
import subprocess
import threading
import queue
import yt_dlp
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import asyncio
import io
import imageio_ffmpeg
import numpy as np
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import torch
from pydub import AudioSegment
import boto3

import shutil

# Configure Pydub (Global)
try:
    from pydub import AudioSegment, utils
    import warnings
    
    # 1. Set FFMPEG from imageio
    ffmpeg_exe = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    
    # VERIFY PATH
    if not os.path.exists(ffmpeg_exe):
        print(f"CRITICAL ERROR: FFMPEG binary not found at: {ffmpeg_exe}")
    else:
        print(f"DEBUG: FFMPEG binary confirmed at: {ffmpeg_exe}")

    AudioSegment.converter = ffmpeg_exe
    
    # 2. Add to PATH (Crucial for pydub to find ffprobe/ffmpeg via subprocess)
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + ffmpeg_dir
    
    # 3. Suppress "Couldn't find ffprobe" warning
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub.utils")

    print(f"DEBUG: Pydub configured with FFMPEG: {ffmpeg_exe}")
except Exception as e:
    print(f"WARNING: Could not configure Pydub: {e}") 
    import traceback
    traceback.print_exc()

class NewsStreamer:
    def __init__(self, youtube_url, target_lang='hi'):
        self.youtube_url = youtube_url
        self.target_lang = target_lang
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.translated_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.ffmpeg_exe = AudioSegment.converter # Use the globally configured one

        # Load environment variables for AWS
        self.load_env_credentials()
        
        # Initialize Polly Client
        try:
            self.polly_client = boto3.client(
                'polly',
                region_name=os.environ.get('AWS_REGION_NAME', 'us-east-1'),
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
            )
            print("Amazon Polly Client Initialized.")
        except Exception as e:
            print(f"WARNING: Amazon Polly Init Failed: {e}")
            self.polly_client = None

        # Initialize models
        # Auto-detect GPU (Run on CUDA if available for Colab)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"Loading Whisper Model on {device} ({compute_type})...")
        # Downgrade to small to prevent CUDA OOM with Parler TTS
        self.model = WhisperModel("small" if device == "cuda" else "tiny", device=device, compute_type=compute_type)
        print("Whisper Model Loaded.")

        # Warmup TTS to prevent first-request timeout
        if hasattr(self, 'tts_model'):
             print("Warming up TTS model...")
             try:
                 dummy_text = "Namaste"
                 self.generate_indic_tts(dummy_text, do_sample=False)
                 print("TTS Warmup Complete.")
             except Exception as e:
                 print(f"Warmup warning: {e}")

        # Load Indic Parler-TTS ONLY if GPU is available
        # On CPU, it is too slow (70s+), so we will fallback to EdgeTTS
        self.tts_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            print("Loading Indic Parler-TTS (GPU)...")
            from parler_tts import ParlerTTSForConditionalGeneration
            self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(self.tts_device)
            self.tts_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
            self.description_tokenizer = AutoTokenizer.from_pretrained(self.tts_model.config.text_encoder._name_or_path)
            print("Indic Parler-TTS Loaded.")
        else:
            print("No GPU detected. Skipping Parler-TTS (Will use EdgeTTS fallback).")

        # Cache silence chunk for keep-alive
        self.silence_chunk = self.get_silence_chunk()

    def load_env_credentials(self):
        # Manually load from .env if variables are missing
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            try:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                env_path = os.path.join(base_dir, '.env')
                if os.path.exists(env_path):
                    print(f"Loading credentials from {env_path}")
                    with open(env_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith('#'): continue
                            if '=' in line:
                                key, val = line.split('=', 1)
                                key = key.strip()
                                val = val.strip()
                                if key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION_NAME']:
                                    os.environ[key] = val
            except Exception as e:
                print(f"Error loading .env: {e}")

    def get_audio_stream_url(self):
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(self.youtube_url, download=False)
            return info['url']

    def capture_audio(self):
        """
        Captures audio from the live stream using ffmpeg and chunks it.
        """
        try:
            print(f"Attempting to get audio URL for: {self.youtube_url}")
            stream_url = self.get_audio_stream_url()
        except Exception as e:
            print(f"CRITICAL ERROR getting stream URL: {e}")
            self.stop_event.set()
            return

        # FFMPEG command to read stream and output raw PCM or wav
        command = [
            self.ffmpeg_exe,
            '-y', 
            '-loglevel', 'error', # Back to error only
            '-re', # Read input at native frame rate
            '-i', stream_url,
            '-f', 's16le', # Raw PCM
            '-ac', '1', # Mono
            '-ar', '16000', # 16kHz for Whisper
            'pipe:1'
        ]
        
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**6)
            
            # Reduce latency: 4 seconds chunks (16000 Hz * 2 bytes * 4)
            chunk_size = 16000 * 2 * 4
            
            while not self.stop_event.is_set():
                audio_data = process.stdout.read(chunk_size)
                if not audio_data:
                    break
                self.audio_queue.put(audio_data)

            # Process finished or stopped, check return code and stderr
            stdout, stderr = process.communicate()
            if process.returncode != 0 and stderr:
                 print(f"FFMPEG Error: {stderr.decode()}")

        except Exception as e:
            print(f"Error in FFMPEG loop: {e}")
        finally:
            if 'process' in locals():
                try:
                    process.terminate()
                except:
                    pass
            print("Audio capture finished.")

    def transcribe_worker(self):
        while not self.stop_event.is_set():
            try:
                audio_data = self.audio_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Convert raw bytes to float32 numpy array for whisper
            # Data is s16le (16-bit signed little-endian)
            # We need to normalize to [-1, 1]
            try:
                # Create numpy array from bytes
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                segments, _ = self.model.transcribe(
                    audio_array, 
                    beam_size=5,
                    vad_filter=True,
                    condition_on_previous_text=False
                )
                text = " ".join([segment.text for segment in segments]).strip()
                
                # Filter out common Whisper hallucinations
                blocklist = ["Thanks for watching!", "Thanks for watching.", "You", "you"]
                if text in blocklist:
                    print(f"Ignored Hallucination: {text}")
                    continue
                    
                if text:
                    print(f"Transcribed: {text}")
                    # Remove repetitions from previous
                    self.text_queue.put(text)
            except Exception as e:
                print(f"Transcription Error: {e}")

    def translate_worker(self):
        translator = GoogleTranslator(source='auto', target=self.target_lang)
        while not self.stop_event.is_set():
            try:
                text = self.text_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            try:
                translated = translator.translate(text)
                print(f"Translated: {translated}")
                self.translated_queue.put(translated)
            except Exception as e:
                print(f"Translation Error: {e}")



    def get_wav_header(self, sample_rate=44100):
        # Generate a standard WAV header with a very large file size (simulating infinite stream)
        import struct
        
        bits_per_sample = 16
        channels = 1
        byte_rate = sample_rate * channels * 2
        block_align = channels * 2
        data_size = 2**31 - 1 # Max 32-bit integer (~2GB audio)
        
        header = b'RIFF'
        header += struct.pack('<I', 36 + data_size)
        header += b'WAVEfmt '
        header += struct.pack('<I', 16) # Subchunk1Size
        header += struct.pack('<H', 1)  # AudioFormat (1=PCM)
        header += struct.pack('<H', channels)
        header += struct.pack('<I', sample_rate)
        header += struct.pack('<I', byte_rate)
        header += struct.pack('<H', block_align)
        header += struct.pack('<H', bits_per_sample)
        header += b'data'
        header += struct.pack('<I', data_size)
        
        return header

    def generate_indic_tts(self, text, do_sample=False):
        import numpy as np
        print(f"DEBUG: Starting TTS Gen on {self.tts_device} for text len: {len(text)}")
        
        try:
            # Description (Voice Style) - Fixed from user preference
            description = """
            Jaya's voice high-pitched, engaging voice is captured in a clear, close-sounding recording.
            His slightly slower delivery conveys a ANGER tone.
            """
            
            # Tokenize description with its own tokenizer
            input_ids = self.description_tokenizer(description, return_tensors="pt").input_ids.to(self.tts_device)
            
            # Tokenize text content with the TTS tokenizer
            prompt_input_ids = self.tts_tokenizer(text, return_tensors="pt").input_ids.to(self.tts_device)
            
            print("DEBUG: Calling model.generate...")
            t0 = time.time()
            generation = self.tts_model.generate(
                input_ids=input_ids, 
                prompt_input_ids=prompt_input_ids,
                do_sample=do_sample
            )
            print(f"DEBUG: Generation done in {time.time()-t0:.2f}s. Processing audio...")
            audio_arr = generation.cpu().numpy().squeeze()
            
            # 1. Normalize Audio (Boost to max volume)
            max_val = np.abs(audio_arr).max()
            if max_val > 0:
                audio_arr = audio_arr / max_val
                # Reduce slightly to 0.95 to avoid clipping
                audio_arr = audio_arr * 0.95
                
            # 2. Convert to Int16
            audio_int16 = (audio_arr * 32767).astype(np.int16)
            
            # 3. Convert to MP3 Bytes (Standard for Streaming)
            import io
            import soundfile as sf
            from pydub import AudioSegment
            
            # Check if we need to flatten (Mono)
            if len(audio_int16.shape) > 1:
                audio_int16 = audio_int16.flatten()

            wav_buffer = io.BytesIO()
            print(f"DEBUG: Config Sample Rate: {self.tts_model.config.sampling_rate}")
            # Explicitly use 44100Hz to match standard, preventing 'vibrator' (slow/deep) perception if config differs.
            sf.write(wav_buffer, audio_int16, 44100, format='WAV', subtype='PCM_16')
            wav_buffer.seek(0)
            
            # Load and Convert
            print("DEBUG: Converting WAV to MP3...")
            seg = AudioSegment.from_wav(wav_buffer)
            
            # Standardize: 44.1kHz, Mono (Good Quality MP3)
            if seg.frame_rate != 44100:
                seg = seg.set_frame_rate(44100)
            if seg.channels != 1:
                seg = seg.set_channels(1)
                
            # Boost Volume
            seg = seg + 10 
            
            # Return MP3 Bytes 
            mp3_buffer = io.BytesIO()
            seg.export(mp3_buffer, format="mp3", bitrate="192k")
            val = mp3_buffer.getvalue()
            print(f"DEBUG: Audio Size - PCM: {len(seg.raw_data)} | MP3: {len(val)}")
            return val
        except Exception as e:
            print(f"CRITICAL TTS ERROR: {e}")
            import traceback
            traceback.print_exc()
            return b""



    def tts_worker(self):
        # Synchronous worker (Polly is sync)
        while not self.stop_event.is_set():
            try:
                text = self.translated_queue.get(timeout=1)
            except queue.Empty:
                continue
                
            try:
                indic_langs = ['hi', 'bn', 'mr', 'ta', 'te', 'ml', 'kn', 'gu', 'pa', 'or', 'as', 'ur', 'ne', 'sd', 'si']
                
                # Check if we should use Parler (Must be Indic AND Model Loaded [GPU])
                use_parler = (self.target_lang in indic_langs) and hasattr(self, 'tts_model')

                audio_bytes = b""
                if use_parler:
                    print(f"Generating IndicTTS (PCM/Parler)...")
                    audio_bytes = self.generate_indic_tts(text)
                    
                    # FALLBACK: If IndicTTS failed or returned empty, use Polly
                    if not audio_bytes:
                        print("WARNING: IndicTTS returned empty/failed. Fallback to Polly.")
                        audio_bytes = self.generate_tts(text)
                else:
                    audio_bytes = self.generate_tts(text)
                
                if audio_bytes:
                    print(f"DEBUG: TTS Generated {len(audio_bytes)} bytes for: {text[:30]}...")
                    self.tts_queue.put(audio_bytes)
                else:
                    print(f"WARNING: TTS generated 0 bytes (Even after fallback) for: {text[:30]}...")
                    
            except Exception as e:
                print(f"TTS Worker Error: {e}")
                import traceback
                traceback.print_exc()



    def generate_tts(self, text):
        if not self.polly_client:
            print("CRITICAL: Polly Client not initialized. Cannot generate TTS.")
            return b""

        # Voice Selection
        # Hindi -> Aditi (Neural)
        # English -> Joanna (Neural)
        # Others -> Fallback to Aditi (if Indic context preferred) or Joanna.
        voice_id = 'Joanna' # Default
        if self.target_lang == 'hi':
            voice_id = 'Aditi'
        # Basic mapping for others if supported by Polly, otherwise fallback
        # Polly supports: hi-IN, ta-IN (boto3 documentation checks needed, but assuming Aditi/Kajal/etc.)
        # Aditi is bilingual (Hindi/English). Kajal is Hindi.
        
        try:
            print(f"DEBUG: Calling Polly ({voice_id}) for text: {text[:30]}...")
            try:
                response = self.polly_client.synthesize_speech(
                    Text=text,
                    OutputFormat='mp3',
                    VoiceId=voice_id,
                    Engine='neural'
                )
            except self.polly_client.exceptions.EngineNotSupportedException as e:
                # specific exception check if possible, otherwise generic client error
                print(f"WARNING: Neural engine not supported for {voice_id}. Falling back to Standard.")
                response = self.polly_client.synthesize_speech(
                    Text=text,
                    OutputFormat='mp3',
                    VoiceId=voice_id,
                    Engine='standard'
                )
            except Exception as e:
                if "does not support the selected engine" in str(e):
                    print(f"WARNING: Neural engine error for {voice_id}. Falling back to Standard. Error: {e}")
                    response = self.polly_client.synthesize_speech(
                        Text=text,
                        OutputFormat='mp3',
                        VoiceId=voice_id,
                        Engine='standard'
                    )
                else:
                    raise e
            
            if 'AudioStream' in response:
                mp3_data = response['AudioStream'].read()
                
                # Standardize to 44.1kHz using FFMPEG
                # Polly Neural is typically 24kHz
                try:
                    cmd = [
                        self.ffmpeg_exe,
                        '-y',
                        '-i', 'pipe:0',
                        '-ar', '44100',
                        '-ac', '1',
                        '-b:a', '192k',
                        '-f', 'mp3',
                        'pipe:1'
                    ]
                    
                    process = subprocess.Popen(
                        cmd, 
                        stdin=subprocess.PIPE, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE
                    )
                    
                    out, err = process.communicate(input=mp3_data)
                    
                    if process.returncode != 0:
                        print(f"CRITICAL: FFMPEG resampling failed: {err.decode()}")
                        return b""
                        
                    return out
                except Exception as e:
                    print(f"FFMPEG Resample Error: {e}")
                    return b""
            else:
                print("Polly returned no AudioStream")
                return b""

        except Exception as e:
            print(f"Polly Synthesis Error: {e}")
            return b""

    def start_streaming(self):
        t1 = threading.Thread(target=self.capture_audio, daemon=True)
        t2 = threading.Thread(target=self.transcribe_worker, daemon=True)
        t3 = threading.Thread(target=self.translate_worker, daemon=True)
        t4 = threading.Thread(target=self.tts_worker, daemon=True)
        
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        
        return [t1, t2, t3, t4]

    def stop(self):
        print("Stopping Streamer...")
        self.stop_event.set()
        
    # Valid 1s Silence MP3 (44.1kHz, Mono, 64kbps - minimal valid frame)
    SILENCE_MP3_B64 = "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU3LjgzLjEwMAAAAAAAAAAAAAAA//oeZVAAAAAAAAAAAAAATGFtZTMuOTkuNVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//oeZRABi4AAAAANIAAAAAExhbWUzLjk5LjVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    
    def get_silence_chunk(self):
        # 1. Try Pydub
        try:
            from pydub import AudioSegment
            import io
            silence = AudioSegment.silent(duration=1000, frame_rate=44100)
            silence = silence.set_channels(1).set_sample_width(2)
            buf = io.BytesIO()
            silence.export(buf, format="mp3", bitrate="192k")
            return buf.getvalue()
        except:
            pass
        
        # 2. Try Direct FFMPEG
        try:
            cmd = [
                self.ffmpeg_exe, '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=mono', '-t', '1', '-f', 'mp3', '-'
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, input=b'')
            return result.stdout
        except:
            pass
            
        # 3. Final Fallback: Hardcoded Base64
        import base64
        return base64.b64decode(self.SILENCE_MP3_B64)

    def audio_generator(self):
        self.start_streaming()
        
        # 1. Yield silence IMMEDIATELY
        yield self.silence_chunk
        
        print("Waiting for audio chunks...")
        try:
            # 2. Initial Loop (Robust)
            while not self.stop_event.is_set():
                try:
                    # Wait for TTS audio
                    chunk = self.tts_queue.get(timeout=2) 
                    yield chunk
                except queue.Empty:
                    if self.stop_event.is_set(): break
                    # Send keep-alive silence
                    yield self.silence_chunk
        except GeneratorExit:
            print("Client disconnected.")
        except Exception as e:
            print(f"Stream Error: {e}")
        finally:
            self.stop()
