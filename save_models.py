# save_models.py (Updated Version)
import torch
import os
from speechbrain.pretrained import EncoderClassifier

print("This script will download and prepare the necessary AI models.")
input("Press Enter to continue...")

try:
    os.makedirs("models", exist_ok=True)

    # --- 1. Save Silero VAD model (Robust Method) ---
    print("\n[1/3] Downloading Silero VAD model...")
    # Load the PyTorch model first
    model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

    # Create a dummy input tensor for the ONNX export.
    # The model expects a 1D tensor of audio samples.
    dummy_input = torch.randn(1024)

    print("      > Converting Silero VAD to ONNX format...")
    torch.onnx.export(model,
                      dummy_input,
                      "models/silero_vad.onnx",
                      opset_version=11,
                      input_names=['input'],
                      output_names=['output'])
    
    print("      > Silero VAD saved to 'models/silero_vad.onnx'")


    # --- 2. Save Speaker Embedding model (ECAPA-TDNN) ---
    print("\n[2/3] Downloading ECAPA-TDNN speaker embedding model...")
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    # Define a dummy input for the model export
    dummy_input_emb = torch.randn(1, 16000)

    print("      > Converting ECAPA-TDNN to ONNX format...")
    torch.onnx.export(classifier.mods.embedding_model,
                      dummy_input_emb,
                      "models/ecapa_tdnn.onnx",
                      opset_version=11,
                      input_names=['input_wav'],
                      output_names=['embedding'])

    print("      > ECAPA-TDNN saved to 'models/ecapa_tdnn.onnx'")
    print("\n[3/3] Model setup complete!")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please ensure you have an internet connection and try again.")