from typing import Dict, List, Type
from omegaconf import DictConfig

import h5py
import hydra

import pytorch_lightning as pl
import torch


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        image: torch.Tensor,
        segment: torch.Tensor,
        overlap: torch.Tensor,
        split: str,
    ) -> None:
        super().__init__()
        image_ht, image_wd, image_ch = cfg.image_shape
        assert image.shape[0] == segment.shape[0] == overlap.shape[0]
        assert image.shape[1] == segment.shape[1] == overlap.shape[1] >= cfg.num_views_all >= cfg.num_views_max >= \
            cfg.num_views_min >= 1
        assert image.shape[2:] == (image_ht, image_wd, image_ch)
        assert segment.shape[2:] == (image_ht, image_wd)
        assert overlap.shape[2:] == (image_ht, image_wd)
        self.cfg = cfg
        self.data_slots = cfg.data_slots[split]
        self.permute = (split == 'train')
        if 'num_views_data' in self.cfg and not self.permute:
            image = image[:, :self.cfg.num_views_data]
            segment = segment[:, :self.cfg.num_views_data]
            overlap = overlap[:, :self.cfg.num_views_data]
        if cfg.num_views_all > 1:
            self.image = image      # [B, V, H, W, C],  uint8
            self.segment = segment  # [B, V, H, W],     uint8
            self.overlap = overlap  # [B, V, H, W],     uint8
        else:
            self.image = image.contiguous().view(-1, 1, *image.shape[2:])        # [B * V, 1, H, W, C],  uint8
            self.segment = segment.contiguous().view(-1, 1, *segment.shape[2:])  # [B * V, 1, H, W],     uint8
            self.overlap = overlap.contiguous().view(-1, 1, *overlap.shape[2:])  # [B * V, 1, H, W],     uint8

    def __len__(self) -> int:
        return self.image.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = (self.image[idx].to(torch.float32) / 255) * 2 - 1
        segment = self.segment[idx][:, None, ..., None].to(torch.int64)
        segment = torch.zeros([segment.shape[0], self.data_slots, *segment.shape[2:]]).scatter_(1, segment, 1)
        overlap = torch.gt(self.overlap[idx][:, None, ..., None], 1).to(torch.float32)
        batch = {
            'image': image,      # [V, H, W, C]
            'segment': segment,  # [V, K, H, W, 1]
            'overlap': overlap,  # [V, 1, H, W, 1]
        }
        if self.permute:
            indices = torch.randperm(image.shape[0])[:self.cfg.num_views_all]
            batch = {key: val[indices] for key, val in batch.items()}
        else:
            batch = {key: val[:self.cfg.num_views_all] for key, val in batch.items()}
        return batch


class DummyDataset(BaseDataset):
    def __init__(self, cfg: DictConfig, split: str) -> None:
        image_ht, image_wd, image_ch = cfg.image_shape
        num_views = cfg.num_views_data
        num_data = cfg.batch_size[split] * 10
        image = torch.randint(
            high=8,
            size=[num_data, num_views, image_ht, image_wd, image_ch],
            dtype=torch.uint8,
        )
        segment = torch.randint(
            high=cfg.data_slots[split],
            size=[num_data, num_views, image_ht, image_wd],
            dtype=torch.uint8,
        )
        overlap = torch.randint(
            high=cfg.data_slots[split],
            size=[num_data, num_views, image_ht, image_wd],
            dtype=torch.uint8,
        )
        super().__init__(cfg, image, segment, overlap, split)


class CustomDataset(BaseDataset):
    def __init__(self, cfg: DictConfig, split: str) -> None:
        alternative_dict = {
            'train':   ['training'],
            'val':     ['validation', 'valid'],
            'test':    ['testing', 'test_1'],
            'general': ['generalization', 'test_2'],
        }
        with h5py.File(hydra.utils.to_absolute_path(cfg.path), 'r') as f:
            if split in f:
                f_key = split
            else:
                for f_key in alternative_dict[split]:
                    if f_key in f:
                        break
                else:
                    raise KeyError
            image = torch.tensor(f[f_key]['image'][()])       # [B, V, H, W, C] or [B, H, W, C],  uint8
            segment = torch.tensor(f[f_key]['segment'][()])   # [B, V, H, W]    or [B, H, W],     uint8
            if 'overlap' in f[f_key]:
                overlap = torch.tensor(f[f_key]['overlap'][()])  # [B, V, H, W] or [B, H, W],     uint8
            else:
                overlap = torch.zeros_like(segment)              # [B, V, H, W] or [B, H, W],     uint8
        if image.ndim == 4 and segment.ndim == 3 and overlap.ndim == 3:
            image = self.image[:, None]      # [B, V, H, W, C],  uint8
            segment = self.segment[:, None]  # [B, V, H, W],     uint8
            overlap = self.overlap[:, None]  # [B, V, H, W],     uint8
        super().__init__(cfg, image, segment, overlap, split)


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, cls_dataset: Type) -> None:
        super().__init__()
        self.cfg = cfg.dataset
        self.cls_dataset = cls_dataset
        self.split_size = torch.cuda.device_count()

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            split_list = ['train', 'val']
        elif stage == 'validate':
            split_list = ['val']
        elif stage == 'predict':
            split_list = ['test', 'general']
        else:
            raise NotImplementedError
        self.data = {split: self.cls_dataset(self.cfg, split) for split in split_list}
        return

    def get_local_batch_size(self, batch_size: int) -> int:
        assert batch_size % self.split_size == 0
        local_batch_size = batch_size // self.split_size
        return local_batch_size

    def train_dataloader(self) -> torch.utils.data.dataloader.DataLoader:
        split = 'train'
        dataloader = torch.utils.data.DataLoader(
            self.data[split],
            batch_size=self.get_local_batch_size(self.cfg.batch_size[split]),
            num_workers=self.cfg.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self) -> torch.utils.data.dataloader.DataLoader:
        split = 'val'
        dataloader = torch.utils.data.DataLoader(
            self.data[split],
            batch_size=self.get_local_batch_size(self.cfg.batch_size[split]),
            num_workers=self.cfg.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def predict_dataloader(self) -> List[torch.utils.data.dataloader.DataLoader]:
        split_list = ['test', 'general']
        dataloader_list = [
            torch.utils.data.DataLoader(
                self.data[split],
                batch_size=self.get_local_batch_size(self.cfg.batch_size[split]),
                num_workers=self.cfg.num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )
            for split in split_list
        ]
        return dataloader_list


class DummyDataModule(BaseDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, DummyDataset)


class CustomDataModule(BaseDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, CustomDataset)
