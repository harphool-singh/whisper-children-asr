import json
import os
from pathlib import Path
from collections import Counter
import statistics

# Paths
TRANSCRIPT_PATH = "data/transcripts/train_word_transcripts.jsonl"
AUDIO_DIR = "data/audio"

print("=" * 60)
print("DATA EXPLORATION")
print("=" * 60)

# Load all transcripts
utterances = []
with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        utterances.append(json.loads(line.strip()))

print(f"\n✅ Total utterances: {len(utterances):,}")

# Preview first 3 utterances
print("\n📋 Sample utterances:")
print("-" * 60)
for utt in utterances[:3]:
    print(f"  ID       : {utt['utterance_id']}")
    print(f"  Child ID : {utt['child_id']}")
    print(f"  Age      : {utt['age_bucket']}")
    print(f"  Duration : {utt['audio_duration_sec']:.2f}s")
    print(f"  Text     : {utt['orthographic_text']}")
    print("-" * 60)

# Age distribution
print("\n👶 Age Distribution:")
age_counts = Counter(utt["age_bucket"] for utt in utterances)
for age, count in sorted(age_counts.items()):
    percent = (count / len(utterances)) * 100
    print(f"  {age:10s}: {count:6,} utterances ({percent:.1f}%)")

# Audio duration stats
durations = [utt["audio_duration_sec"] for utt in utterances]
print(f"\n⏱️  Audio Duration Stats:")
print(f"  Min      : {min(durations):.2f}s")
print(f"  Max      : {max(durations):.2f}s")
print(f"  Average  : {statistics.mean(durations):.2f}s")
print(f"  Median   : {statistics.median(durations):.2f}s")
total_hours = sum(durations) / 3600
print(f"  Total    : {total_hours:.2f} hours")

# Transcript length stats
word_counts = [len(utt["orthographic_text"].split()) for utt in utterances]
print(f"\n📝 Transcript Word Count Stats:")
print(f"  Min words    : {min(word_counts)}")
print(f"  Max words    : {max(word_counts)}")
print(f"  Avg words    : {statistics.mean(word_counts):.1f}")

# Check missing audio files
print(f"\n🔍 Checking missing audio files...")
missing = 0
for utt in utterances:
    audio_path = os.path.join(AUDIO_DIR, utt["audio_path"])
    if not os.path.exists(audio_path):
        missing += 1
print(f"  Missing files: {missing}")
print(f"  Found files  : {len(utterances) - missing:,}")

print("\n" + "=" * 60)
print("EXPLORATION COMPLETE!")
print("=" * 60)

# Duration filtering analysis
too_long = 0
too_short = 0
good = 0

for utt in utterances:
    dur = utt["audio_duration_sec"]
    if dur > 30:
        too_long += 1
    elif dur < 0.5:
        too_short += 1
    else:
        good += 1

print(f"\n📊 Duration Filtering Analysis:")
print(f"  Too long  (>30s) : {too_long} ({too_long/len(utterances)*100:.1f}%)")
print(f"  Too short (<0.5s): {too_short} ({too_short/len(utterances)*100:.1f}%)")
print(f"  Good clips       : {good} ({good/len(utterances)*100:.1f}%)")
