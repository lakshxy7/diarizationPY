# convert_vad.py
import torch
from speechbrain.pretrained import VAD
import os

OUTPUT_FOLDER = "models/"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_FOLDER, "vad.onnx")

print("--- Starting SpeechBrain VAD to ONNX Conversion ---")

try:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("Downloading and loading SpeechBrain VAD model...")
    vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")
    vad.mods['model'].eval()

    print("Creating dummy input for export...")
    dummy_input = torch.randn(1, 16000)  # 1 second of audio at 16kHz

    print(f"Exporting model to ONNX at '{OUTPUT_FILE_PATH}'...")
    torch.onnx.export(
        vad.mods['model'],
        dummy_input,
        OUTPUT_FILE_PATH,
        opset_version=11,
        input_names=['wav'],
        output_names=['speech_prob']
    )

    print("\nConversion successful!")
    print(f"Model saved to '{OUTPUT_FILE_PATH}'")

except Exception as e:
    print(f"\nAn error occurred during conversion: {e}")
    print("This may be another deep PyTorch/ONNX bug.")
