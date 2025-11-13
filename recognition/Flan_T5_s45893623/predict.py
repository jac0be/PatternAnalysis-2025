# predict.py
import argparse
import torch
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

DEFAULT_PROMPT = (
    "You are a helpful medical assistant. Rewrite the radiology report for a layperson "
    "in 1–3 sentences, avoid jargon, use plain language.\n\n"
    "Report:\n{rad_report}\n\nLayperson summary:"
)

# Loads the model checkpoint and does prediction
@torch.no_grad()
def predict(report_text, ckpt_dir="runs/flan_t5_lora", prompt=None, beams=4, max_new=128):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir).to(dev).eval()

    p = (prompt or DEFAULT_PROMPT).format(rad_report=report_text)
    enc = tok([p], return_tensors="pt", truncation=True, max_length=1024).to(dev)
    out = model.generate(
        **enc,
        max_new_tokens=max_new,
        num_beams=beams,
        early_stopping=True
    )
    return tok.batch_decode(out, skip_special_tokens=True)[0]

# Loads up an interactive check. Uses same default run dir as train.py. If idx is set, it computes the summary for that val_report[idx] and exits.
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="runs/flan_t5_lora",
                   help="Checkpoint directory")
    p.add_argument("--idx", type=int, default=None,
                   help="Index of val-set report to evaluate")
    args = p.parse_args()

    # If idx is provided: run prediction on that test report
    if args.idx is not None:
        ds = load_dataset(
            "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
        )["validation"]

        report = ds[args.idx]["radiology_report"]
        gold = ds[args.idx]["layman_report"]
        print(f"\n--- Test Sample {args.idx} ---")
        print("Radiology Report:\n", report, "\n")
        print("Gold Summary:\n", gold, "\n")
        pred = predict(report, ckpt_dir=args.ckpt)
        print("Model Prediction:\n", pred, "\n")
        return

    # Otherwise: interactive chat
    print(f"Chat with your FLAN-T5 model ({args.ckpt}). Please enter only the report you want summarised. Type 'exit' to quit.\n")
    while True:
        msg = input("You: ").strip()
        if msg.lower() in {"exit", "quit"}:
            break
        reply = predict(msg, ckpt_dir=args.ckpt)
        print("Model:", reply, "\n")

if __name__ == "__main__":
    main()
