import json
import torch
import librosa
import jiwer
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm

# Paths
MODEL_PATH = "submission/model"
VAL_JSONL = "data/transcripts/val_split.jsonl"
AUDIO_DIR = "data/audio"

# Settings
NUM_SAMPLES = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000

print("=" * 60)
print("BASELINE EVALUATION - WHISPER LARGE V3")
print("=" * 60)
print(f"\n🖥️  Device  : {DEVICE}")
print(f"📊 Samples : {NUM_SAMPLES}")

# Load model
print("\n📥 Loading model...")
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
)
model = model.to(DEVICE)

model.eval()
print("✅ Model loaded!")

# Load validation data
utterances = []
with open(VAL_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        utterances.append(json.loads(line.strip()))

# Use only first N samples
utterances = utterances[:NUM_SAMPLES]
print(f"📂 Loaded {len(utterances)} validation samples")

# Run inference
references = []
hypotheses = []

print("\n🔄 Running inference...")
for utt in tqdm(utterances):
    try:
        # Load audio
        audio_path = Path(AUDIO_DIR) / utt["audio_path"]
        audio, _ = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)

        # Process
        inputs = processor(
    audio,
    sampling_rate=TARGET_SR,
    return_tensors="pt"
).input_features.to(DEVICE, dtype=torch.float16)


        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(
                inputs,
                language="en",
                task="transcribe"
            )

        # Decode
        predicted_text = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0].strip().lower()

        references.append(utt["orthographic_text"].lower())
        hypotheses.append(predicted_text)

    except Exception as e:
        print(f"⚠️ Error on {utt['utterance_id']}: {e}")
        continue

# Calculate WER
wer = jiwer.wer(references, hypotheses)


print(f"\n📊 BASELINE RESULTS:")
print(f"  Samples evaluated : {len(references)}")
print(f"  WER               : {wer:.4f} ({wer*100:.2f}%)")

# Show some examples
print(f"\n📝 Sample Predictions:")
print("-" * 60)
for i in range(5):
    print(f"  Reference  : {references[i]}")
    print(f"  Predicted  : {hypotheses[i]}")
    print("-" * 60)

print("\n" + "=" * 60)
print("BASELINE EVALUATION COMPLETE!")
print("=" * 60)
