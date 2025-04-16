from torch.utils.data import Dataset, DataLoader
from typing import List
from PIL import Image


# Custom Dataset
class ImageDataset(Dataset):
    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return image

