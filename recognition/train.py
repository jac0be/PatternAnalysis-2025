# train.py
# flan t5 base + LORA training
import os, math, time, argparse, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import evaluate # for rouge scoring

# Our codebase
from dataset import BioSummDataset
from modules import load_tokenizer, build_flan_t5_with_lora

PROMPT = (
    "You are a helpful medical assistant. Rewrite the radiology report for a layperson "
    "in 1–3 sentences, avoid jargon, use plain language.\n\n"
    "Report:\n{rad_report}\n\nLayperson summary:"
)

def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="google/flan-t5-base")
    p.add_argument("--out_dir", default="runs/flan_t5_lora")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_input_len", type=int, default=1024)
    p.add_argument("--max_target_len", type=int, default=256)
    p.add_argument("--val_beams", type=int, default=4)
    p.add_argument("--val_max_new_tokens", type=int, default=128)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class Batchify:

    def __init__(self, tok, max_in, max_out):
        self.tok = tok
        self.max_in = max_in
        self.max_out = max_out
        self.pad_id = tok.pad_token_id

    def __call__(self, batch):
        src = [PROMPT.format(rad_report=x) for x, _ in batch] # include radiology report in prompt
        tgt = [y for _, y in batch]
        # encode src and tgt seperately so we can mask
        enc = self.tok(src, padding=True, truncation=True, max_length=self.max_in, return_tensors="pt")
        dec = self.tok(text_target=tgt, padding=True, truncation=True, max_length=self.max_out, return_tensors="pt")
        labels = dec["input_ids"]
        # pad tokens get -100 so loss ignores them.
        labels[labels == self.pad_id] = -100
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

@torch.no_grad()
def score_rouge(model, tok, loader, dev, max_new_tokens, beams):

    if loader is None: return None
    model.eval()
    preds, refs = [], []

    for b in loader:
        b = {k: v.to(dev) for k, v in b.items()}
        gen = model.generate(
            input_ids=b["input_ids"],
            attention_mask=b["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=beams,
            early_stopping=True # only 1-3 max sentences anyway
        )

        # put the pads back so decode handles -100s
        tgt = b["labels"].clone()
        tgt[tgt == -100] = tok.pad_token_id
        preds.extend(tok.batch_decode(gen, skip_special_tokens=True))
        refs.extend(tok.batch_decode(tgt, skip_special_tokens=True))

    r = evaluate.load("rouge")
    out = r.compute(predictions=preds, references=refs, use_stemmer=True)

    return {k: float(out[k]) for k in ("rouge1","rouge2","rougeL","rougeLsum") if k in out}

def run_one_epoch(model, loader, optim, sched, scaler, dev, accum, use_amp, log_every=50, step_hook=None):

    model.train()
    total, shown, t0 = 0.0, 0.0, time.time()
    steps = 0

    for i, batch in enumerate(loader, 1):

        batch = {k: v.to(dev) for k, v in batch.items()}

        with autocast(enabled=use_amp):
            out = model(**batch)
            # gradient accumulation
            loss = out.loss / accum

        scaler.scale(loss).backward()
        total += loss.item()
        shown += loss.item()

        if i % accum == 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # prevents wild spikes
            scaler.step(optim); scaler.update()
            optim.zero_grad(set_to_none=True)

            if sched is not None: sched.step()
            steps += 1

            # we call model peak here
            if step_hook is not None and (steps % 500 == 0):
                step_hook(steps)

            if steps % log_every == 0:
                dt = time.time() - t0
                mean_loss = shown / log_every
                print({"step": steps, "loss": round(mean_loss, 6), "sec": round(dt, 2)})
                shown = 0.0; t0 = time.time()

    return total / max(1, steps)

# we peak at the model every 500 steps & ask it to generate a summary of the first report in the dataset.
# this is only to visually confirm that the model is improving its summaries.
@torch.no_grad()
def model_peak(model, tok, dev, dataset, beams=4, max_new=128):
    
    model.eval()
    x, y = dataset[0]
    enc = tok([PROMPT.format(rad_report=x)], return_tensors="pt", truncation=True, max_length=1024).to(dev)

    out = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=max_new,
        num_beams=beams,
        early_stopping=True
    )
    pred = tok.batch_decode(out, skip_special_tokens=True)[0]

    print("---------MODEL_PEAK----------")
    print(f"Radiology Report: \n{x}")
    print(f"True: \n {y}")
    print(f"LLM: \n {pred}")
    print("----------------------------")

def main():

    a = args_parse()
    os.makedirs(a.out_dir, exist_ok=True)
    set_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = load_tokenizer(a.model_name)
    model = build_flan_t5_with_lora(
        model_name=a.model_name, r=a.lora_r, alpha=a.lora_alpha, dropout=a.lora_dropout
    )
    model.config.use_cache = False
    model.to(dev)

    train_ds = BioSummDataset(split="train")
    val_ds   = BioSummDataset(split="validation")
    test_ds  = BioSummDataset(split="test")

    collate = Batchify(tok, a.max_input_len, a.max_target_len)
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate)



    optim = AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd) # adam@ optimiser
    total_updates = math.ceil(len(train_loader) / max(1, a.grad_accum)) * a.epochs # updates =  round_up(batches per epoch /accum) * epochs
    warm = min(a.warmup_steps, max(1, total_updates // 20))
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warm, num_training_steps=total_updates)
    scaler = GradScaler(enabled=(a.fp16 and dev == "cuda"))

    # model peak prior to training
    model_peak(model, tok, dev, train_ds, beams=a.val_beams, max_new=a.val_max_new_tokens)

    # we pass this _probe as a function reference to our epoch trainer, runs every 500 steps.
    def _probe(_step):
        model_peak(model, tok, dev, train_ds, beams=a.val_beams, max_new=a.val_max_new_tokens)

    best = -1.0
    for ep in range(1, a.epochs + 1):
        print(f"\nepoch {ep}/{a.epochs}")
        tr_loss = run_one_epoch(
            model, train_loader, optim, sched, scaler, dev,
            a.grad_accum, a.fp16, log_every=a.grad_accum, step_hook=_probe # <- here
        )
        print({"train_loss": round(float(tr_loss), 6)})

        scores = score_rouge(model, tok, val_loader, dev, a.val_max_new_tokens, a.val_beams)
        if scores:
            msg = {k: round(v, 4) for k, v in scores.items()}
            print({"val": msg})
            cur = scores.get("rougeLsum", scores.get("rougeL", -1.0))
            if cur > best:
                best = cur
                model.save_pretrained(a.out_dir)
                tok.save_pretrained(a.out_dir)
                # logging + print to show which epoch got saved
                with open(os.path.join(a.out_dir, "best.json"), "w", encoding="utf-8") as f:
                    f.write(str({"epoch": ep, "metric": cur}))
                print({"save": a.out_dir, "metric": round(cur, 4)})

    # Eval: Validation
    final_val = score_rouge(model, tok, val_loader, dev, a.val_max_new_tokens, a.val_beams)
    print({"final_val": {k: round(v, 4) for k, v in final_val.items()}})
    # Eval: Test
    final_test = score_rouge(model, tok, test_loader, dev, a.val_max_new_tokens, a.val_beams)
    print({"final_test": {k: round(v, 4) for k, v in final_test.items()}})
    with open(os.path.join(a.out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        # we log for test
        import json
        json.dump({k: float(v) for k, v in final_test.items()}, f, indent=2)


if __name__ == "__main__":
    main()
