import sys
from pathlib import Path
from typing import Callable, Dict, List, Union

import torch
from torch.utils.data import Dataset

from datasets import (
    CUB,
    INaturalist2021,
    Places365,
    ROxfordRParisDataset,
    StanfordOnlineProducts,
)

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(1, str(PROJECT_ROOT))


def get_dataset(dataroot: Union[str, Path], dataset_name: str, split: str, preprocess: Callable, **kwargs) -> Dataset:
    dataset_name = dataset_name.lower()
    dataroot = Path(dataroot)

    if dataset_name == "roxford5k":
        return ROxfordRParisDataset(dataroot, "roxford5k", split, preprocess=preprocess)
    if dataset_name == "rparis6k":
        return ROxfordRParisDataset(dataroot, "rparis6k", split, preprocess=preprocess)
    if dataset_name == "cub2011":
        return CUB(dataroot, split, preprocess=preprocess)
    if dataset_name == "sop":
        return StanfordOnlineProducts(dataroot, split, preprocess=preprocess)
    if dataset_name == "inaturalist2021":
        return INaturalist2021(dataroot, split, preprocess=preprocess)
    if dataset_name == "places365":
        return Places365(dataroot, split, preprocess=preprocess)
    if dataset_name == "stanford_cars":
        from datasets.dassl.stanford_cars import StanfordCars

        return StanfordCars(dataroot, split, preprocess=preprocess)
    if dataset_name == "oxford_pets":
        from datasets.dassl.oxford_pets import OxfordPets

        return OxfordPets(dataroot, split, preprocess=preprocess)
    if dataset_name == "oxford_flowers":
        from datasets.dassl.oxford_flowers import OxfordFlowers

        return OxfordFlowers(dataroot, split, preprocess=preprocess)
    if dataset_name == "fgvc_aircraft":
        from datasets.dassl.fgvc_aircraft import FGVCAircraft

        return FGVCAircraft(dataroot, split, preprocess=preprocess)
    if dataset_name == "dtd":
        from datasets.dassl.dtd import DescribableTextures

        return DescribableTextures(dataroot, split, preprocess=preprocess)
    if dataset_name == "eurosat":
        from datasets.dassl.eurosat import EuroSAT

        return EuroSAT(dataroot, split, preprocess=preprocess)
    if dataset_name == "food101":
        from datasets.dassl.food101 import Food101

        return Food101(dataroot, split, preprocess=preprocess)
    if dataset_name == "sun397":
        from datasets.dassl.sun397 import SUN397

        return SUN397(dataroot, split, preprocess=preprocess)
    if dataset_name == "caltech101":
        from datasets.dassl.caltech101 import Caltech101

        return Caltech101(dataroot, split, preprocess=preprocess)
    if dataset_name == "ucf101":
        from datasets.dassl.ucf101 import UCF101

        return UCF101(dataroot, split, preprocess=preprocess)
    if dataset_name == "imagenet":
        from datasets.dassl.imagenet import ImageNet

        return ImageNet(dataroot, split, preprocess=preprocess)

    raise ValueError(f"Dataset {dataset_name} is not enabled in this image-to-image retrieval project")


def collate_fn(batch: List[Union[torch.Tensor, None]]) -> torch.Tensor:
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


RETRIEVAL_SPLTS: Dict[str, Dict[str, str]] = {
    "cub2011": {"query": "all", "gallery": "all"},
    "roxford5k": {"query": "query", "gallery": "gallery"},
    "rparis6k": {"query": "query", "gallery": "gallery"},
    "sop": {"query": "test", "gallery": "test"},
    "inaturalist2021": {"query": "train", "gallery": "train"},
    "places365": {"query": "val", "gallery": "val"},
    "caltech101": {"query": "test", "gallery": "train"},
    "dtd": {"query": "test", "gallery": "train"},
    "eurosat": {"query": "test", "gallery": "train"},
    "fgvc_aircraft": {"query": "test", "gallery": "train"},
    "food101": {"query": "test", "gallery": "train"},
    "imagenet": {"query": "test", "gallery": "train"},
    "oxford_flowers": {"query": "test", "gallery": "train"},
    "oxford_pets": {"query": "test", "gallery": "train"},
    "stanford_cars": {"query": "test", "gallery": "train"},
    "sun397": {"query": "test", "gallery": "train"},
    "ucf101": {"query": "test", "gallery": "train"},
}
