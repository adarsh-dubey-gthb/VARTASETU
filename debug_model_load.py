import sys
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

model_name = "ai4bharat/indic-parler-tts"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")
try:
    print("Loading Model...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Model Loaded Successfully!")
    print(f"Sampling Rate: {model.config.sampling_rate}")
    
except Exception as e:
    print(f"Failed to load model: {e}")
    import traceback
    traceback.print_exc()
