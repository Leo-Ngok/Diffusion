import os
from jittor.dataset import DataLoader, ImageFolder
from jittor.transform import Compose, Resize, RandomResizedCrop, ToTensor, ImageNormalize
from jittor.misc import make_grid
import jittor as jt

from argparse import Namespace
from PIL import Image

def setup_logging(run_name: str) -> None:
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def get_data(args: Namespace):
    transforms = Compose([
        Resize(256),  # args.image_size + 1/4 *args.image_size
        RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        ToTensor(),
        ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def save_images(images: jt.Var, path: str, **kwargs):
    grid:jt.Var = make_grid(images, **kwargs)
    assert type(grid) is jt.Var
    ndarr_trp:jt.Var = grid.permute(1, 2, 0)
    ndarr = ndarr_trp.numpy()
    im = Image.fromarray(ndarr)
    im.save(path)
