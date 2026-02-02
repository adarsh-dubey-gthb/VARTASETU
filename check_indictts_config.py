from transformers import AutoConfig

model_name = "ai4bharat/indic-parler-tts"
try:
    import transformers
    print(f"Transformers Version: {transformers.__version__}")
    print(f"Transformers Path: {transformers.__file__}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print(f"Model: {model_name}")
    print(f"Sampling Rate: {getattr(config, 'sampling_rate', 'NOT FOUND')}")
    print(f"Config: {config}")
except Exception as e:
    print(f"Error loading config: {e}")
