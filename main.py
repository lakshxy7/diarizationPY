# ~/Desktop/diarizationPY/main.py
import onnxruntime as ort
import sounddevice as sd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import collections
import time
import sys
import torch
import torchaudio

# --- Configuration & Constants ---
VAD_MODEL_PATH = "models/silero_vad.jit" 
EMBED_MODEL_PATH = "models/ecapa_tdnn.onnx"
333
SAMPLE_RATE = 16000
BLOCKSIZE = 512
VOICE_BUFFER_SECONDS = 2
VAD_CONFIDENCE_THRESHOLD = 0.5
SIMILARITY_THRESHOLD = 0.65

VOICE_BUFFER_SIZE = int(VOICE_BUFFER_SECONDS * SAMPLE_RATE / BLOCKSIZE)

# --- Global State Variables ---
voice_buffer = collections.deque(maxlen=VOICE_BUFFER_SIZE)
is_speaking = False
known_speakers = {}
speaker_count = 0
vad_model = None
embed_session = None

# --- Main Functions ---

def load_models():
    """Loads the VAD model (PyTorch) and Embedding model (ONNX)."""
    global vad_model, embed_session
    
    print(f"Loading VAD model (PyTorch JIT) from: {VAD_MODEL_PATH}")
    vad_model = torch.jit.load(VAD_MODEL_PATH)
    vad_model.eval()

    print(f"Loading Speaker Embedding model (ONNX) from: {EMBED_MODEL_PATH}")
    embed_session = ort.InferenceSession(EMBED_MODEL_PATH)
    print("Models loaded successfully.")

def audio_callback(indata, frames, time, status):
    """This function is called for each new audio chunk from the microphone."""
    global is_speaking, speaker_count, voice_buffer, known_speakers
    
    if status:
        print(status, file=sys.stderr)

    vad_input_numpy = indata.flatten().astype(np.float32)
    vad_input_tensor = torch.from_numpy(vad_input_numpy)
    
    speech_prob = vad_model(vad_input_tensor, torch.tensor(SAMPLE_RATE)).item()
    
    is_speech = speech_prob > VAD_CONFIDENCE_THRESHOLD

    if is_speech:
        if not is_speaking:
            is_speaking = True
            voice_buffer.clear()
            print("Speech detected, recording utterance...", end="", flush=True)
        voice_buffer.append(indata)
    else:
        if is_speaking:
            is_speaking = False
            print(" done.")
            
            if len(voice_buffer) < 30:
                print("Utterance too short, ignoring.")
                return

            full_utterance = np.concatenate(list(voice_buffer))
            
            # --- Feature Extraction and Speaker ID ---
            utterance_tensor = torch.from_numpy(full_utterance.flatten().astype(np.float32))

            # --- THIS IS THE CORRECTED SECTION ---
            # The argument is 'sample_frequency', not 'sample_rate'.
            mel_spectrogram = torchaudio.compliance.kaldi.fbank(
                waveform=utterance_tensor.unsqueeze(0),
                sample_frequency=SAMPLE_RATE,
                num_mel_bins=80,
            )
            
            features_numpy = mel_spectrogram.numpy()
            new_embedding = embed_session.run(None, {'feats': features_numpy})[0]

            found_speaker_id = "Unknown Speaker"
            if len(known_speakers) > 0:
                similarities = {sid: cosine_similarity(new_embedding, emb)[0][0] for sid, emb in known_speakers.items()}
                max_sim_id = max(similarities, key=similarities.get)
                max_sim_value = similarities[max_sim_id]
                
                if max_sim_value > SIMILARITY_THRESHOLD:
                    found_speaker_id = max_sim_id
            
            if found_speaker_id == "Unknown Speaker":
                speaker_count += 1
                found_speaker_id = f"Speaker_{speaker_count}"
                known_speakers[found_speaker_id] = new_embedding

            print(f"--- Utterance from: {found_speaker_id} ---\n")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        load_models()
        print("\nStarting audio stream... Speak into your microphone. Press Ctrl+C to stop.")
        
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE, dtype='float32'):
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping application.")
    except Exception as e:
        print(f"An error occurred: {e}")