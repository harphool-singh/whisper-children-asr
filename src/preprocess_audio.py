import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import json

AUDIO_DIR      = "data/audio"
PROCESSED_DIR  = "data/audio_16k"
TRANSCRIPT     = "data/transcripts/train_word_transcripts.jsonl"
TARGET_SR      = 16000

print("=" * 60)
print("  PRE-PROCESSING AUDIO TO 16kHz MONO")
print("=" * 60)

# Load all utterance paths
utterances = []
with open(TRANSCRIPT, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            utterances.append(json.loads(line))

print(f"\n📂 Total files : {len(utterances):,}")
print(f"📁 Output dir  : {PROCESSED_DIR}")

# Create output directory
Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

# Process each file
skipped   = 0
converted = 0
errors    = 0

for utt in tqdm(utterances, desc="Converting"):
    src_path = Path(AUDIO_DIR) / utt["audio_path"]
    dst_path = Path(PROCESSED_DIR) / utt["audio_path"]

    # Skip if already processed
    if dst_path.exists():
        skipped += 1
        continue

    # Create subdirectory if needed
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Load and resample
        audio, _ = librosa.load(str(src_path), sr=TARGET_SR, mono=True)
        # Save as 16kHz mono FLAC
        sf.write(str(dst_path), audio, TARGET_SR)
        converted += 1
    except Exception as e:
        errors += 1

print(f"\n✅ Converted : {converted:,}")
print(f"⏭️  Skipped   : {skipped:,}")
print(f"❌ Errors    : {errors:,}")
print("\n✅ Pre-processing complete!")
