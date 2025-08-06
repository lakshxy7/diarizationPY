# Assets/model_converter.py
# Final version with 'aug' argument fix for export.
import torch
import os
import sys

# --- Configuration ---
REPO_PATH = "ECAPA-TDNN/"
WEIGHTS_FILE = os.path.join(REPO_PATH, "exps/pretrain.model")
OUTPUT_FOLDER = "../models/"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_FOLDER, "ecapa_tdnn.onnx")

# --- Main Conversion Logic ---
def main():
    """Loads the PyTorch model and converts it to ONNX."""
    
    print("--- Starting ECAPA-TDNN to ONNX Conversion ---")

    if not os.path.isdir(REPO_PATH):
        print(f"ERROR: Repository folder not found at '{REPO_PATH}'")
        return

    if not os.path.exists(WEIGHTS_FILE):
        print(f"ERROR: Weights file not found at '{WEIGHTS_FILE}'")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    try:
        sys.path.append(REPO_PATH)
        from model import ECAPA_TDNN
        sys.path.pop()
    except ImportError:
        print(f"ERROR: Could not import 'ECAPA_TDNN' from '{os.path.join(REPO_PATH, 'model.py')}'.")
        return
        
    print("Loading model architecture...")
    model = ECAPA_TDNN(C=1024)
    model.eval()

    print(f"Loading and cleaning weights from '{WEIGHTS_FILE}'...")
    original_state_dict = torch.load(WEIGHTS_FILE, map_location=torch.device('cpu'))
    cleaned_state_dict = {}
    prefix_to_remove = "speaker_encoder."
    for k, v in original_state_dict.items():
        if k.startswith(prefix_to_remove):
            new_key = k[len(prefix_to_remove):]
            cleaned_state_dict[new_key] = v
    model.load_state_dict(cleaned_state_dict)
    print("Weights loaded successfully!")

    dummy_input = torch.randn(1, 16000)
    
    print(f"Exporting model to ONNX at '{OUTPUT_FILE_PATH}'...")
    
    # --- THIS IS THE CORRECTED LINE ---
    # We now pass a tuple of arguments: (dummy_input, False) to satisfy the 'aug' requirement.
    torch.onnx.export(model,
                      (dummy_input, False),
                      OUTPUT_FILE_PATH,
                      opset_version=11,
                      input_names=['input_wav', 'aug'], # Also name the new input
                      output_names=['embedding'],
                      dynamic_axes={'input_wav': {1: 'audio_length'}})

    print("\nConversion successful!")
    print(f"Model saved to '{OUTPUT_FILE_PATH}'")


if __name__ == "__main__":
    main()