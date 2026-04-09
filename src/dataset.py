import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class ChildrenSpeechDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        audio_dir,
        processor,
        noise_dir=None,
        noise_prob=0.0,
        target_sr=16000,
    ):
        self.utterances = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.utterances.append(json.loads(line))

        self.feature_dir = Path(audio_dir)
        self.processor   = processor

        print(f"✅ Dataset ready: {len(self.utterances):,} samples")
        print(f"   Features dir: {self.feature_dir}")

        # Quick check first file exists
        first = self.utterances[0]
        test_path = self.feature_dir / first["audio_path"].replace(".flac", ".npy")
        if test_path.exists():
            arr = np.load(str(test_path))
            print(f"   Feature shape: {arr.shape} ✅")
        else:
            print(f"❌ Feature file not found: {test_path}")

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt       = self.utterances[idx]
        feat_path = (
            self.feature_dir
            / utt["audio_path"].replace(".flac", ".npy")
        )
        # Instant load — pure disk I/O
        mel    = np.load(str(feat_path)).astype(np.float32)
        labels = self.processor.tokenizer(
            utt["orthographic_text"]
        ).input_ids

        return {"input_features": mel, "labels": labels}
