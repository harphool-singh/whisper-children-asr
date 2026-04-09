import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

NOISE_DIR  = "data/noise"
TARGET_SR  = 16000

noise_files = list(Path(NOISE_DIR).rglob("*.flac"))
print(f"Found {len(noise_files)} noise files")
print("Resampling all noise files to 16kHz...")

already_good = 0
resampled    = 0

for f in tqdm(noise_files):
    audio, sr = librosa.load(str(f), sr=None, mono=True)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sf.write(str(f), audio, TARGET_SR)
        resampled += 1
    else:
        already_good += 1

print(f"\n✅ Already 16kHz : {already_good}")
print(f"✅ Resampled     : {resampled}")
print("✅ All noise files are now 16kHz!")
