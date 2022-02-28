from typing import List, Optional

import numpy as np
from skimage.color import rgb2lab
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid


class ColorizationDataset(Dataset):
    def __init__(
        self, path_list: List[str], split: str = "train", image_size: int = 256
    ):
        self.split = split
        self.path_list = path_list

        if split == "train":
            self.transforms = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size), Image.BICUBIC),
                    transforms.RandomHorizontalFlip(),  # horizontal flip makes no sense for this task i think
                ]
            )
        elif split == "val":
            self.transforms = transforms.Resize((image_size, image_size), Image.BICUBIC)

    def __getitem__(self, idx):
        img = np.array(
            self.transforms(Image.open(self.path_list[idx]).convert("RGB"))
        )  # to avoid mutation
        img_lab = transforms.ToTensor()(rgb2lab(img).astype("float32"))
        L = img_lab[[0], ...] / 50.0 - 1.0  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.0  # Between -1 and 1

        return {"L": L, "ab": ab}

    def __len__(self):
        return len(self.paths)


class ColorizationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: List[str], batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ColorizationDataset(path_list=self.data_dir, split="train")
        self.val_dataset = ColorizationDataset(path_list=self.data_dir, split="val")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
