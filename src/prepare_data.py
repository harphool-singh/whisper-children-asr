import json
import random
from pathlib import Path

# Paths
TRANSCRIPT_PATH = "data/transcripts/train_word_transcripts.jsonl"
OUTPUT_DIR = "data/transcripts"

# Settings
MIN_DURATION = 0.5
MAX_DURATION = 30.0
VAL_SPLIT = 0.10
RANDOM_SEED = 42

print("=" * 60)
print("DATA FILTERING & SPLITTING")
print("=" * 60)

# Load all utterances
utterances = []
with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        utterances.append(json.loads(line.strip()))

print(f"\n📂 Total utterances loaded : {len(utterances):,}")

# Filter utterances
filtered = []
for utt in utterances:
    dur = utt["audio_duration_sec"]
    if MIN_DURATION <= dur <= MAX_DURATION:
        filtered.append(utt)

print(f"✅ After filtering          : {len(filtered):,}")
print(f"❌ Removed                  : {len(utterances) - len(filtered):,}")

# Shuffle before splitting
random.seed(RANDOM_SEED)
random.shuffle(filtered)

# Split into train and validation
val_size = int(len(filtered) * VAL_SPLIT)
train_size = len(filtered) - val_size

train_data = filtered[val_size:]
val_data = filtered[:val_size]

print(f"\n📊 Split Results:")
print(f"  Training set   : {len(train_data):,} utterances (90%)")
print(f"  Validation set : {len(val_data):,} utterances (10%)")

# Save train split
train_path = Path(OUTPUT_DIR) / "train_split.jsonl"
with open(train_path, "w", encoding="utf-8") as f:
    for utt in train_data:
        f.write(json.dumps(utt) + "\n")
print(f"\n💾 Saved training split   : {train_path}")

# Save validation split
val_path = Path(OUTPUT_DIR) / "val_split.jsonl"
with open(val_path, "w", encoding="utf-8") as f:
    for utt in val_data:
        f.write(json.dumps(utt) + "\n")
print(f"💾 Saved validation split : {val_path}")

# Age distribution in each split
from collections import Counter

train_ages = Counter(utt["age_bucket"] for utt in train_data)
val_ages = Counter(utt["age_bucket"] for utt in val_data)

print(f"\n👶 Age Distribution in Training:")
for age, count in sorted(train_ages.items()):
    print(f"  {age:10s}: {count:,}")

print(f"\n👶 Age Distribution in Validation:")
for age, count in sorted(val_ages.items()):
    print(f"  {age:10s}: {count:,}")

print("\n" + "=" * 60)
print("FILTERING & SPLITTING COMPLETE!")
print("=" * 60)
