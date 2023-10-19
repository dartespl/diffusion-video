"""
Train a diffusion model on images.
"""
# import sys
# import os
# sys.path.append("C:/Users/its/Documents/repos/guided-diffusion")

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import torch

from guided_diffusion.image_datasets import ImageDataset, _list_image_files_recursively, MPI
from torch.utils.data import DataLoader

from guided_diffusion.gaussian_diffusion import _extract_into_tensor

import numpy as np

from guided_diffusion.nn import mean_flat
from guided_diffusion.losses import normal_kl, discretized_gaussian_log_likelihood

from torch.utils.data import Dataset
import neptune
from neptune.types import File


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    run = neptune.init_run(
        project="dartespl/diffusion-video",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5OGMwNjU0Ny1lY2Q5LTRiZWItODU4ZS1mYWRiYTU2MTYxODUifQ==",
    )  # your credentials

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=5,
        image_size=args.image_size,
        deterministic=False,
        random_flip=False
    )

    batch, _ = next(data)
    t = torch.tensor([1,30,60,90,120])
    print(t, t-1)
    frames = q_sample_frames(batch, t)
    save_image_to_neptune(run, frames, "lb")
    # for i,frame in enumerate(frames):
        # save_image_to_neptune(run, frame, "lb")
        # print(frame)
        # from torchvision import utils
        # frame = ((frame + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        # # frame = frame.permute(1, 2, 0)
        # utils.save_image(frame, str(t[i])+".jpg")

    run.stop()

import sys
import argparse

import cv2
print(cv2.__version__)


def save_image_to_neptune(run, img, label):
    img = ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    img = img.permute(0, 2, 3, 1)
    img = img.contiguous()
    logger.log(img.shape)

    run["image_series"].append(
        File.as_image(
                img[0].cpu() / 255
            ),  # You can upload arrays as images using the File.as_image() method
            name=label,
        )

def q_sample_frames(paths, t):
    transformed = []
    for i in range(t.shape[0]):
        from guided_diffusion import logger
        if not paths[i]:
            continue
        # logger.log(paths[0], t[0])
        frame = extractFrame(paths[i], t[i].item())
        if frame is None:
            continue
        # logger.log(frame[0])
        img = transformImage(frame)
        # logger.log(img[0])
        transformed.append(img)

    from guided_diffusion import dist_util
    return torch.tensor(transformed).to(device=dist_util.dev())

def transformImage(img):
    from guided_diffusion.image_datasets import center_crop_arr
    img = torch.tensor(img)
    img = np.transpose(img, [2, 0, 1])
    import torchvision
    # print(img.shape)
    img = torchvision.transforms.functional.to_pil_image(img)
    # print(img.shape)
    arr = center_crop_arr(img, 32)
    arr = arr.astype(np.float32) / 127.5 - 1
    return np.transpose(arr, [2, 0, 1])

def extractFrame(pathIn, frame_num):
    import cv2
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    seconds = int(frames / fps)

    vidcap.set(cv2.CAP_PROP_POS_MSEC,(frame_num * (seconds*8)))    # added this line 
    success,image = vidcap.read()
    if image is None:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,0)    # added this line 
        success,image = vidcap.read()
    return image

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    dataset = VideoDataset(
        all_files,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False
        )
    logger.log("xdd")
    while True:
        yield from loader


class VideoDataset(Dataset):
    def __init__(
        self,
        video_paths,
    ):
        super().__init__()
        self.local_videos = video_paths

    def __len__(self):
        return len(self.local_videos)

    def __getitem__(self, idx):
        path = self.local_videos[idx]
        return path, 0


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,

        use_neptune=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
