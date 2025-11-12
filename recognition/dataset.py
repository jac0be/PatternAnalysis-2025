# Simple dataset handler
from datasets import load_dataset
from torch.utils.data import Dataset

class BioSummDataset(Dataset):
    def __init__(self, split="train"):
        self.ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")[split]

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