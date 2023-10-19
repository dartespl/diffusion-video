"""
Train a diffusion model on images.
"""
# import sys
# import os
# sys.path.append("C:/Users/its/Documents/repos/guided-diffusion")

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data, _list_image_files_recursively
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

import neptune
from neptune.types import File

from torch.utils.data import Dataset, DataLoader


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    run = neptune.init_run(
        project="dartespl/diffusion-video",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5OGMwNjU0Ny1lY2Q5LTRiZWItODU4ZS1mYWRiYTU2MTYxODUifQ==",
    )  # your credentials

    # logger.log("creating model and diffusion...")
    # model, diffusion = create_model_and_diffusion(
    #     **args_to_dict(args, model_and_diffusion_defaults().keys())
    # )
    # model.load_state_dict(
    #     dist_util.load_state_dict(args.model_path, map_location="cpu")
    # )
    # model.to(dist_util.dev())
    # if args.use_fp16:
    #     model.convert_to_fp16()
    # model.eval()
    # schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=1,
        image_size=128,
        class_cond=False,
        deterministic=True
    )
    paths, cond = next(data)
    # path = paths[0]


    ###############
    model_kwargs = None
    shape = (1,3,128,128)
    progress = False

    device = dist_util.dev()
    assert isinstance(shape, (tuple, list))
    # if noise is not None:
    #     img = noise
    # else:
    #     img = th.randn(*shape, device=device)
    # imgs = q_sample_last_frames(paths).to(device)
    # img = imgs[0]
    # img = th.randn(*shape, device=device)
    indices = list(range(128))[::-1]


    # logger.log(imgs.shape, shape)

    if progress:
        # Lazy import so that we don't depend on tqdm.
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    # for i in indices:
    imgs = q_sample_frames(paths, torch.tensor(indices)).to(device)
    save_image_to_neptune(run, imgs, f"image step {22}")
    ################

    run.stop()


def save_image_to_neptune(run, img, label):
    img = ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    img = img.permute(0, 2, 3, 1)
    img = img.contiguous()
    logger.log(img.shape)

    for i in range(128):
        run["image_series"].append(
            File.as_image(
                    img[i].cpu() / 255
                ),  # You can upload arrays as images using the File.as_image() method
                name=f"image step {i}",
        )
def q_sample_frames(paths, t):
        transformed = []
        for i in range(t.shape[0]):
            from guided_diffusion import logger
            # if not paths[i]:
            #     continue
            # logger.log(paths[0], t[0])
            frame = extractFrame(paths[0], t[i].item())
            # if frame is None:
            #     continue
            # logger.log(frame[0])
            img = transformImage(frame)
            # logger.log(img[0])
            transformed.append(img)

        # from . import dist_util
        return torch.tensor(transformed)

def transformImage(img):
    from guided_diffusion.image_datasets import center_crop_arr
    img = torch.tensor(img.copy())
    img = np.transpose(img, [2, 0, 1])
    import torchvision
    img = torchvision.transforms.functional.to_pil_image(img)
    # print(img.shape)
    
    # print(img.shape)
    arr = center_crop_arr(img, 128)
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
    return image[..., ::-1]

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
        model_path="",
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

        clip_denoised=True,
        num_samples=10,
        use_ddim=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()