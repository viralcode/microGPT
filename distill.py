"""
Knowledge Distillation for microGPT
======================================
Train a smaller (student) model to mimic a larger (teacher) model.
The student learns to match the teacher's output probability distribution
using KL divergence, producing a smaller model with much of the teacher's
capability.

This is the same technique used by:
  - DeepSeek R1 Distill (671B → 7B/14B/70B)
  - Qwen distillation series
  - DistilBERT, TinyLLaMA

How it works:
  1. Teacher model produces soft probability distributions (logits)
  2. Student model is trained to match those distributions (KL divergence)
  3. Optionally, student also trains on ground-truth tokens (cross-entropy)
  4. Temperature scaling softens distributions for better knowledge transfer

Usage:
    # Distill a large model into a small one:
    python distill.py --teacher checkpoints/large.pt --student-preset small \\
                      --data data/ --temperature 2.0

    # Distill with custom loss balance (alpha=0.7 means 70% distillation, 30% CE):
    python distill.py --teacher checkpoints/large.pt --student-preset small \\
                      --data data/ --alpha 0.7 --temperature 3.0

    # Resume distillation from a student checkpoint:
    python distill.py --teacher large.pt --student checkpoints/student.pt \\
                      --data data/

Reference:
    Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
"""

import os
import sys
import math
import time
import pickle
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig, get_model_config
from model import GPT


def load_data(data_dir: str, split: str, block_size: int, batch_size: int, device: str):
    """Load a batch of training data."""
    data_path = os.path.join(data_dir, f"{split}.bin")

    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        dtype = np.uint32 if meta.get("tokenizer_type") == "tiktoken" else np.uint16
    else:
        dtype = np.uint16

    data = np.memmap(data_path, dtype=dtype, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def distillation_loss(student_logits, teacher_logits, targets,
                      temperature=2.0, alpha=0.5):
    """Compute combined distillation + cross-entropy loss.

    Args:
        student_logits: (B, T, V) student model output
        teacher_logits: (B, T, V) teacher model output (detached)
        targets: (B, T) ground truth token IDs
        temperature: softens distributions (higher = softer, more knowledge)
        alpha: balance between distillation and CE (1.0 = pure distillation)

    Returns:
        loss: combined loss
        kd_loss: distillation component (for logging)
        ce_loss: cross-entropy component (for logging)
    """
    # KL Divergence on soft targets (distillation loss)
    # Scale logits by temperature before softmax
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    # KL(teacher || student) — summed over vocab, averaged over batch
    kd_loss = F.kl_div(
        student_soft.view(-1, student_soft.size(-1)),
        teacher_soft.view(-1, teacher_soft.size(-1)),
        reduction="batchmean",
    )
    # Scale by T^2 (Hinton et al.) to keep gradients balanced
    kd_loss = kd_loss * (temperature ** 2)

    # Standard cross-entropy on hard targets
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        targets.view(-1),
        ignore_index=-1,
    )

    # Combined loss
    loss = alpha * kd_loss + (1 - alpha) * ce_loss

    return loss, kd_loss.item(), ce_loss.item()


@torch.no_grad()
def evaluate(student, teacher, data_dir, block_size, batch_size,
             device, temperature, alpha, n_iters=50):
    """Evaluate distillation loss on validation set."""
    student.eval()
    teacher.eval()
    losses = []
    for _ in range(n_iters):
        X, Y = load_data(data_dir, "val", block_size, batch_size, device)
        student_logits, _ = student(X)
        teacher_logits, _ = teacher(X)
        loss, _, _ = distillation_loss(
            student_logits, teacher_logits, Y,
            temperature=temperature, alpha=alpha,
        )
        losses.append(loss.item())
    student.train()
    return sum(losses) / len(losses)


def distill(args):
    """Main distillation training loop."""

    # -- Device --
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # -- Load teacher model (frozen) --
    if not os.path.exists(args.teacher):
        print(f"Error: Teacher checkpoint not found: {args.teacher}")
        sys.exit(1)

    print(f"Loading teacher model: {args.teacher}")
    teacher_ckpt = torch.load(args.teacher, map_location=device, weights_only=False)
    teacher_config = GPTConfig(**teacher_ckpt["model_config"])
    teacher = GPT(teacher_config)
    teacher.load_state_dict(teacher_ckpt["model"])
    teacher.to(device)
    teacher.eval()

    # Freeze teacher completely
    for p in teacher.parameters():
        p.requires_grad = False

    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher: {teacher_params/1e6:.1f}M params (frozen)")

    # -- Create or load student model --
    if args.student:
        # Resume from student checkpoint
        print(f"Loading student model: {args.student}")
        student_ckpt = torch.load(args.student, map_location=device, weights_only=False)
        student_config = GPTConfig(**student_ckpt["model_config"])
        student = GPT(student_config)
        student.load_state_dict(student_ckpt["model"])
    else:
        # Create new student from preset
        print(f"Creating student model: preset={args.student_preset}")
        student_config = get_model_config(args.student_preset)
        # Match teacher's vocab size
        student_config.vocab_size = teacher_config.vocab_size
        student = GPT(student_config)

    student.to(device)
    student.train()

    student_params = sum(p.numel() for p in student.parameters())
    compression = teacher_params / student_params
    print(f"  Student: {student_params/1e6:.1f}M params (trainable)")
    print(f"  Compression: {compression:.1f}x")

    # Use the smaller block size
    block_size = min(teacher_config.block_size, student_config.block_size)

    # -- Mixed precision --
    if "cuda" in device and torch.cuda.is_bf16_supported():
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif "cuda" in device:
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        ctx = nullcontext()

    # -- Optimizer --
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    # -- Gradient checkpointing for student if requested --
    if args.gradient_checkpointing:
        student.enable_gradient_checkpointing()
        print("  Gradient checkpointing: ON")

    # -- Training --
    print(f"\n{'='*60}")
    print(f"  Knowledge Distillation")
    print(f"{'='*60}")
    print(f"  Teacher:     {teacher_params/1e6:.1f}M params")
    print(f"  Student:     {student_params/1e6:.1f}M params ({compression:.1f}x smaller)")
    print(f"  Temperature: {args.temperature}")
    print(f"  Alpha:       {args.alpha} (KD={args.alpha:.0%}, CE={1-args.alpha:.0%})")
    print(f"  LR:          {args.lr}")
    print(f"  Max iters:   {args.max_iters}")
    print(f"  Device:      {device}")
    print(f"{'='*60}\n")

    t0 = time.time()
    best_val_loss = float("inf")
    ema_loss = None
    ema_kd = None
    ema_ce = None

    for iter_num in range(args.max_iters):
        # Cosine LR with warmup
        if iter_num < args.warmup_iters:
            lr = args.lr * iter_num / max(1, args.warmup_iters)
        else:
            progress = (iter_num - args.warmup_iters) / (args.max_iters - args.warmup_iters)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # -- Evaluate --
        if iter_num % args.eval_interval == 0:
            val_loss = evaluate(
                student, teacher, args.data, block_size,
                args.batch_size, device, args.temperature, args.alpha,
            )
            t1 = time.time()
            print(f"  Step {iter_num:>5d} | val loss: {val_loss:.4f} | "
                  f"lr: {lr:.2e} | {t1-t0:.1f}s")
            t0 = t1

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(args.output_dir, exist_ok=True)
                ckpt = {
                    "model": student.state_dict(),
                    "model_config": student_config.to_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "distillation": {
                        "teacher_checkpoint": args.teacher,
                        "temperature": args.temperature,
                        "alpha": args.alpha,
                        "compression": compression,
                    },
                }
                save_path = os.path.join(args.output_dir, "distilled_best.pt")
                torch.save(ckpt, save_path)
                print(f"    Saved best: {save_path}")

            student.train()

        # -- Train step --
        X, Y = load_data(args.data, "train", block_size, args.batch_size, device)

        with ctx:
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits, _ = teacher(X)
                # Need full sequence logits from teacher
                if teacher_logits.size(1) == 1:
                    # Teacher was in inference mode, re-run in train-like mode
                    teacher_logits = teacher.lm_head(
                        teacher.transformer.ln_f(X)
                    )

            # Student forward
            student_logits, _ = student(X)

            # Distillation loss
            # Use full sequence logits
            s_logits = student_logits if student_logits.size(1) > 1 else student_logits
            t_logits = teacher_logits if teacher_logits.size(1) > 1 else teacher_logits

            loss, kd_l, ce_l = distillation_loss(
                s_logits, t_logits, Y,
                temperature=args.temperature, alpha=args.alpha,
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        # EMA tracking
        batch_loss = loss.item()
        if ema_loss is None:
            ema_loss = batch_loss
            ema_kd = kd_l
            ema_ce = ce_l
        else:
            ema_loss = 0.95 * ema_loss + 0.05 * batch_loss
            ema_kd = 0.95 * ema_kd + 0.05 * kd_l
            ema_ce = 0.95 * ema_ce + 0.05 * ce_l

        if iter_num % 100 == 0 and iter_num > 0:
            print(f"    iter {iter_num:>5d} | loss {ema_loss:.4f} "
                  f"(KD={ema_kd:.4f}, CE={ema_ce:.4f}) | lr {lr:.2e}")

    # -- Final save --
    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, "distilled_final.pt")
    ckpt = {
        "model": student.state_dict(),
        "model_config": student_config.to_dict(),
        "iter_num": args.max_iters,
        "best_val_loss": best_val_loss,
        "distillation": {
            "teacher_checkpoint": args.teacher,
            "temperature": args.temperature,
            "alpha": args.alpha,
            "compression": compression,
        },
    }
    torch.save(ckpt, final_path)

    print(f"\n{'='*60}")
    print(f"  Distillation Complete")
    print(f"{'='*60}")
    print(f"  Teacher:         {teacher_params/1e6:.1f}M params")
    print(f"  Student:         {student_params/1e6:.1f}M params ({compression:.1f}x smaller)")
    print(f"  Best val loss:   {best_val_loss:.4f}")
    print(f"  Final checkpoint: {final_path}")
    print(f"{'='*60}")
    print(f"\nGenerate with: python generate.py --checkpoint {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Knowledge distillation for microGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Distill large model into small:
  python distill.py --teacher checkpoints/large.pt --student-preset small --data data/

  # Custom temperature and alpha:
  python distill.py --teacher large.pt --student-preset small --temperature 3.0 --alpha 0.7

  # Resume from student checkpoint:
  python distill.py --teacher large.pt --student checkpoints/distilled_best.pt --data data/
        """,
    )

    # Models
    parser.add_argument("--teacher", type=str, required=True,
                        help="Path to teacher model checkpoint")
    parser.add_argument("--student", type=str, default=None,
                        help="Path to student checkpoint (to resume distillation)")
    parser.add_argument("--student-preset", type=str, default="small",
                        help="Preset for new student model (default: small)")

    # Distillation
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Distillation temperature (default: 2.0, higher=softer)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="KD vs CE balance: 1.0=pure KD, 0.0=pure CE (default: 0.5)")

    # Training
    parser.add_argument("--data", type=str, default="data",
                        help="Directory with train.bin and val.bin")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Output directory")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--max-iters", type=int, default=5000,
                        help="Max training iterations (default: 5000)")
    parser.add_argument("--warmup-iters", type=int, default=200,
                        help="Warmup iterations (default: 200)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--eval-interval", type=int, default=250,
                        help="Evaluation interval (default: 250)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, cpu")

    args = parser.parse_args()

    if not args.student and not args.student_preset:
        print("Error: Must provide either --student (checkpoint) or --student-preset")
        sys.exit(1)

    distill(args)
