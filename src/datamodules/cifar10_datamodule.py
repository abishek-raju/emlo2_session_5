from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LitCIFAR10DataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = "data/",
                 batch_size: int = 64,
                num_workers: int = 0,
                pin_memory: bool = False,):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

    def get_dataset(self, root_dir, train, transform):
        dataset = CIFAR10(
            root=root_dir,
            train=train,
            transform=transform,
            download=True,
        )
        return dataset

    def train_dataloader(self):
        # transform = T.Compose(
        #     [
        #         T.RandomCrop(32, padding=4),
        #         T.RandomHorizontalFlip(),
        #         T.ToTensor(),
        #         T.Normalize(self.mean, self.std),
        #     ]
        # )
        transform = A.Compose(
            [
                A.Normalize(mean=[0.49139968,0.48215841,0.44653091], std=[0.49139968,0.48215841,0.44653091]),
                # A.Rotate(limit = (-7,7),always_apply = True),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=(-7,7), p=.75),
                A.CoarseDropout (max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=[0.49139968,0.48215841,0.44653091],p = 0.75),
                # A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=[1,1,1], always_apply=True, p=0.5),
                A.HorizontalFlip(p=0.75),
                ToTensorV2()
            ]
        )
        dataset = self.get_dataset(
            root_dir=self.data_dir,
            train=True,
            transform=transform,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.get_dataset(self.data_dir, train=False, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
