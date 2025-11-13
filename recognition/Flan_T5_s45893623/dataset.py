# Simple dataset handler
from datasets import load_dataset
from torch.utils.data import Dataset

class BioSummDataset(Dataset):
    def __init__(self, split="train", do_train_split=False):
        ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")
        # We optionally split the training data to get a held-out test set.
        if do_train_split == True and split in ["train", "test"]: # NOTE: You must construct both train and test using do_train_split=True, otherwise the splits won't be executed for both
            full_train = ds["train"]
            split_ds = full_train.train_test_split(test_size=0.1, seed=42) # keep seed set at 42 to keep splits consistent.

            self.ds = split_ds["train"] if split == "train" else split_ds["test"]
        # Otherwise use the default train,validation,test split in BioSumm (NOTE: default test does not contain layman summary)
        else:
            self.ds = ds[split]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        # we do not care about the image or source, only text
        x = self.ds[i]["radiology_report"]
        y = self.ds[i]["layman_report"]
        return x, y    

# Test main: prints out the first 10 in the dataset    
if __name__ == "__main__":
    train_ds = BioSummDataset(split="train")
    val_ds   = BioSummDataset(split="validation")
    test_ds  = BioSummDataset(split="test")
    for i in range(0, 10):
        print(f"Train [{i}]: {train_ds[i][0]}")
        print(f"Val [{i}]: {val_ds[i][0]}")
        print(f"Test [{i}]: {test_ds[i][0]}")