import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model ai4bharat/indic-parler-tts...")
    model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

    # Text from the screenshot
    prompt = "ஒரு தடவை கூட எதையாவது சொன்னால் நூறு முறை சொன்னது போல் எடுக்கப்படும்."
    
    # Description from the screenshot
    description = """
    Jaya's voice high-pitched, engaging voice is captured in a clear, close-sounding recording.
    His slightly slower delivery conveys a ANGER tone.
    """

    print("Tokenizing inputs...")
    input_ids = description_tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    print("Generating audio...")
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    output_filename = "indic_tts_out.wav"
    print(f"Saving audio to {output_filename}...")
    sf.write(output_filename, audio_arr, model.config.sampling_rate)
    print("Done!")

if __name__ == "__main__":
    main()
