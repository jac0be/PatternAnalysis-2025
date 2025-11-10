# predict.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEFAULT_PROMPT = (
    "You are a helpful medical assistant. Rewrite the radiology report for a layperson "
    "in 1–3 sentences, avoid jargon, use plain language.\n\n"
    "Report:\n{rad_report}\n\nLayperson summary:"
)

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

@torch.no_grad()
def preview(model, tok, dev, report_text, ref=None, prompt=None, beams=4, max_new=128):
    p = (prompt or DEFAULT_PROMPT).format(rad_report=report_text)
    enc = tok([p], return_tensors="pt", truncation=True, max_length=1024).to(dev)
    out = model.generate(**enc, max_new_tokens=max_new, num_beams=beams, early_stopping=True)
    pred = tok.batch_decode(out, skip_special_tokens=True)[0]
    print("---------PREVIEW----------")
    print(f"Report:\n{report_text}\n")
    if ref: print(f"True:\n{ref}\n")
    print(f"Pred:\n{pred}\n--------------------------")
    return pred

def main():
    print("Chat with your FLAN-T5 model. Type 'exit' to quit.\n")
    while True:
        msg = input("You: ").strip()
        if msg.lower() in {"exit", "quit"}:
            break
        reply = predict(msg)
        print("Model:", reply, "\n")

if __name__ == "__main__":
    main()
