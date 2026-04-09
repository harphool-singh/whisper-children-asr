import json
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

# ─────────────────────────────────────────────
AUDIO_DIR   = "data/audio_16k"
FEATURE_DIR = "data/features"
TRANSCRIPT  = "data/transcripts/train_word_transcripts.jsonl"
TARGET_SR   = 16000
N_FFT       = 400
HOP_LENGTH  = 160
N_MELS      = 128

NUM_WORKERS = 16   # 16 of your 24 physical cores
# ─────────────────────────────────────────────

def compute_mel(audio):
    """Whisper-compatible mel spectrogram (variable length, no 30s padding)"""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=0,
        fmax=8000,
        power=2.0,
    )
    # Whisper's exact normalization
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float16)


def process_one(utt):
    src = Path(AUDIO_DIR) / utt["audio_path"]
    dst = Path(FEATURE_DIR) / utt["audio_path"].replace(".flac", ".npy")

    if dst.exists():
        return "skip"

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        audio, sr = sf.read(str(src), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        mel = compute_mel(audio)
        np.save(str(dst), mel)
        return "ok"
    except Exception as e:
        return f"err:{e}"


if __name__ == "__main__":
    # Load utterances
    utterances = []
    with open(TRANSCRIPT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                utterances.append(json.loads(line))

    print("=" * 60)
    print("  PRE-COMPUTING MEL FEATURES")
    print("=" * 60)
    print(f"\n📂 Total files  : {len(utterances):,}")
    print(f"⚙️  Workers      : {NUM_WORKERS}")
    print(f"💾 Output dir   : {FEATURE_DIR}")

    Path(FEATURE_DIR).mkdir(parents=True, exist_ok=True)

    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(process_one, utterances, chunksize=50),
            total=len(utterances),
            desc="Computing"
        ))

    ok   = results.count("ok")
    skip = results.count("skip")
    err  = sum(1 for r in results if r.startswith("err"))
    total_size = sum(
        f.stat().st_size for f in Path(FEATURE_DIR).rglob("*.npy")
    ) / 1e9

    print(f"\n✅ Computed : {ok:,}")
    print(f"⏭️  Skipped  : {skip:,}")
    print(f"❌ Errors   : {err}")
    print(f"💾 Total size: {total_size:.1f} GB")
    print("\n✅ Pre-computation complete!")
