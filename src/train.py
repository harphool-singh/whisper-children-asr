import os
import sys
import numpy as np
import torch
import jiwer
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List
from torch.utils.data import DataLoader
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("src")
from dataset import ChildrenSpeechDataset

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
MODEL_PATH     = "submission/model"
TRAIN_JSONL    = "data/transcripts/train_split.jsonl"
VAL_JSONL      = "data/transcripts/val_split.jsonl"
AUDIO_DIR      = "data/features"
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_DIR = "submission/model"

# ─────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────
NUM_EPOCHS    = 2
BATCH_SIZE    = 4
GRAD_ACCUM    = 16       # effective batch = 16 * 4 = 64
LEARNING_RATE = 1e-5
WARMUP_STEPS  = 500
EVAL_STEPS    = 500
SAVE_STEPS    = 500
LOGGING_STEPS = 50
MAX_FRAMES    = 3000    # Whisper requires exactly 3000
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
#  DATA COLLATOR
# ─────────────────────────────────────────────
@dataclass
class DataCollator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]):
        # Pad/truncate mel features to exactly 3000 frames
        mels = []
        for f in features:
            m = f["input_features"]         # (128, n_frames)
            n = m.shape[1]
            if n >= MAX_FRAMES:
                m = m[:, :MAX_FRAMES]
            else:
                m = np.pad(
                    m,
                    ((0, 0), (0, MAX_FRAMES - n)),
                    constant_values=0.0
                )
            mels.append(m)

        input_tensor = torch.tensor(
            np.stack(mels), dtype=torch.float32
        )

        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        return {"input_features": input_tensor, "labels": labels}


# ─────────────────────────────────────────────
#  EVALUATE
# ─────────────────────────────────────────────
def evaluate(model, val_loader, processor, device):
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Evaluating", leave=False):
            input_features = batch["input_features"].to(device, dtype=torch.bfloat16)
            label_ids      = batch["labels"].clone()

            pred_ids = model.generate(
                input_features,
                language="en",
                task="transcribe",
                max_new_tokens=225,
            )

            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
            label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            references.extend([s.lower().strip() for s in label_str])
            hypotheses.extend([s.lower().strip() for s in pred_str])

    wer = jiwer.wer(references, hypotheses)
    model.train()
    return wer


# ─────────────────────────────────────────────
#  CHECKPOINT
# ─────────────────────────────────────────────
def save_checkpoint(step, optimizer, scheduler, best_wer, model, processor):
    Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
    torch.save({
        "step"      : step,
        "best_wer"  : best_wer,
        "optimizer" : optimizer.state_dict(),
        "scheduler" : scheduler.state_dict(),
    }, f"{CHECKPOINT_DIR}/checkpoint.pt")
    model.save_pretrained(f"{CHECKPOINT_DIR}/model")
    processor.save_pretrained(f"{CHECKPOINT_DIR}/model")


def load_checkpoint(optimizer, scheduler, device):
    ckpt_file = f"{CHECKPOINT_DIR}/checkpoint.pt"
    if Path(ckpt_file).exists():
        ckpt = torch.load(ckpt_file, map_location=device)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        print(f"🔄 Resumed from step {ckpt['step']}, best WER: {ckpt['best_wer']:.4f}")
        return ckpt["step"], ckpt["best_wer"]
    return 0, float("inf")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  WHISPER LARGE V3 — TRAINING")
    print("=" * 60)
    print(f"\n🖥️  GPU  : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   VRAM : {vram:.1f} GB")

    # ── Load Model ──
    print("\n📥 Loading model...")
    processor = WhisperProcessor.from_pretrained(MODEL_PATH)
    model     = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    ).to(DEVICE)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens    = []
    model.config.use_cache          = False
    model.gradient_checkpointing_enable()  # Reduces activation memory

    processor.tokenizer.set_prefix_tokens(language="english", task="transcribe")
    print("✅ Model loaded!")

    # ── Datasets ──
    print("\n📂 Loading datasets...")
    collator = DataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    train_dataset = ChildrenSpeechDataset(
        TRAIN_JSONL, AUDIO_DIR, processor
    )
    val_dataset = ChildrenSpeechDataset(
        VAL_JSONL, AUDIO_DIR, processor
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collator, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8,
        shuffle=False, collate_fn=collator, num_workers=0,
    )
    print(f"✅ Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    # ── Optimizer & Scheduler ──
    steps_per_epoch = len(train_loader) // GRAD_ACCUM
    total_steps     = steps_per_epoch * NUM_EPOCHS

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE,
        weight_decay=0.01, fused=True,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, WARMUP_STEPS, total_steps
    )

    start_step, best_wer = load_checkpoint(optimizer, scheduler, DEVICE)

    print(f"\n⚙️  Steps/epoch : {steps_per_epoch:,}")
    print(f"   Total steps : {total_steps:,}")
    print(f"\n🚀 Starting training...")

    # ── Training Loop ──
    global_step  = start_step
    running_loss = 0.0

    model.train()
    optimizer.zero_grad()

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"  EPOCH {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}")
        ):
            input_features = batch["input_features"].to(DEVICE, dtype=torch.bfloat16)
            labels         = batch["labels"].to(DEVICE)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_features=input_features, labels=labels)
                loss    = outputs.loss / GRAD_ACCUM

            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % LOGGING_STEPS == 0:
                    avg  = running_loss / (LOGGING_STEPS * GRAD_ACCUM)

                    lr   = scheduler.get_last_lr()[0]
                    print(f"\n  Step {global_step:5d} | Loss: {avg:.4f} | LR: {lr:.2e}")
                    running_loss = 0.0

                if global_step % EVAL_STEPS == 0:
                    print(f"\n📊 Evaluating at step {global_step}...")
                    wer = evaluate(model, val_loader, processor, DEVICE)
                    print(f"   WER: {wer:.4f} ({wer*100:.2f}%) | Best: {best_wer:.4f}")

                    if wer < best_wer:
                        best_wer = wer
                        model.save_pretrained(BEST_MODEL_DIR)
                        processor.save_pretrained(BEST_MODEL_DIR)
                        print("   ✅ Best model saved!")

                    save_checkpoint(
                        global_step, optimizer, scheduler,
                        best_wer, model, processor
                    )

    print(f"\n{'='*60}")
    print(f"  DONE! Best WER: {best_wer:.4f} ({best_wer*100:.2f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
