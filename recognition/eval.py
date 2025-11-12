# eval.py
# Computes plots and csvs using logged data from the training run.
import os, sys, shutil

def save_curves_and_plots_from_run(run_dir: str):
    """
    Rebuild CSVs and plots using the jsonl logs in a run-like directory (e.g., eval/<run_name>).
    Handles repeated 'step' values by constructing a monotonic 'gstep' and an inferred 'epoch'.
    """
    import csv, json
    import matplotlib.pyplot as plt
    import os

    tl_jsonl = os.path.join(run_dir, "train_loss.jsonl")
    vr_jsonl = os.path.join(run_dir, "val_rouge.jsonl")

    # Load train loss
    raw_loss = []
    if os.path.isfile(tl_jsonl):
        with open(tl_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    raw_loss.append({"step": int(obj.get("step", 0)),
                                     "loss": float(obj.get("loss", 0.0))})
                except Exception:
                    pass

    # Rebuild epoch + global step
    loss_hist = []
    if raw_loss:
        epoch = 1
        prev_step = -1
        carry = 0
        last_epoch_max = 0

        for r in raw_loss:
            s = r["step"]
            # detect wrap (new epoch) when step doesn't increase
            if s <= prev_step:
                carry += max(last_epoch_max, prev_step)
                last_epoch_max = 0
                epoch += 1
            last_epoch_max = max(last_epoch_max, s)
            gstep = carry + s
            loss_hist.append({"epoch": epoch, "step": s, "gstep": gstep, "loss": r["loss"]})
            prev_step = s

    # Load validation rouge
    val_hist = []
    if os.path.isfile(vr_jsonl):
        with open(vr_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    row = {"epoch": int(obj.get("epoch", 0))}
                    for k in ("rouge1", "rouge2", "rougeL", "rougeLsum"):
                        if k in obj:
                            row[k] = float(obj[k])
                    val_hist.append(row)
                except Exception:
                    pass

    # Write CSVs
    if loss_hist:
        loss_csv = os.path.join(run_dir, "train_loss.csv")
        with open(loss_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "step", "gstep", "loss"])
            w.writeheader()
            # already chronological, ensure by gstep
            w.writerows(sorted(loss_hist, key=lambda d: d["gstep"]))

    if val_hist:
        fields = sorted({k for d in val_hist for k in d.keys()})
        val_csv = os.path.join(run_dir, "val_rouge.csv")
        with open(val_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(sorted(val_hist, key=lambda d: d.get("epoch", 0)))

    # Create Plots
    if loss_hist:
        xs = [d["gstep"] for d in loss_hist]
        ys = [d["loss"] for d in loss_hist]
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel("global step"); plt.ylabel("loss"); plt.title("train loss")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "train_loss.png"))
        plt.close()

    if val_hist:
        # single plot, multiple lines (rouge1, rouge2, rougeL, rougeLsum) vs epoch
        epochs = [d.get("epoch", 0) for d in val_hist]
        metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        plt.figure()
        for metric_key in metrics:
            ys = [d.get(metric_key, 0.0) for d in val_hist]
            plt.plot(epochs, ys, marker="o", label=metric_key)
        plt.xlabel("epoch")
        plt.ylabel("ROUGE score")
        plt.title("validation ROUGE over epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "val_rouge.png"))
        plt.close()

def main():
    if len(sys.argv) != 2:
        print("Usage: python eval.py <run_directory>")
        sys.exit(1)

    run_dir = sys.argv[1]
    if not os.path.isdir(run_dir):
        print(f"Error: '{run_dir}' is not a valid directory")
        sys.exit(1)

    run_name = os.path.basename(os.path.normpath(run_dir))
    eval_dir = os.path.join("eval", run_name)

    # create eval/run_name folder, copy jsonls so function can read them
    os.makedirs(eval_dir, exist_ok=True)
    for fname in ("train_loss.jsonl", "val_rouge.jsonl"):
        src = os.path.join(run_dir, fname)
        dst = os.path.join(eval_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    print(f"Rebuilding plots into {eval_dir} ...")
    save_curves_and_plots_from_run(eval_dir)
    print(f"Done. Outputs saved under {eval_dir}")

# Usage: python eval.py runs/flan-t5-lora
if __name__ == "__main__":
    main()
