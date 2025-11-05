# Simple dataset handler
from datasets import load_dataset
from torch.utils.data import Dataset

class BioSummDataset(Dataset):
    def __init__(self, split="train"):
        self.ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        # we do not care about the image or source
        x = self.ds[i]["radiology_report"]
        y = self.ds[i]["layman_report"]
        return x, y



    