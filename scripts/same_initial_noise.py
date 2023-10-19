"""
Train a diffusion model on images.
"""
# import sys
# import os
# sys.path.append("C:/Users/its/Documents/repos/guided-diffusion")

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

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

import neptune
from neptune.types import File
from guided_diffusion.gaussian_diffusion import GaussianDiffusion


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    run = neptune.init_run(
        project="dartespl/diffusion-video",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5OGMwNjU0Ny1lY2Q5LTRiZWItODU4ZS1mYWRiYTU2MTYxODUifQ==",
    )  # your credentials

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # logger.log("creating data loader...")
    # data = load_data(
    #     data_dir=args.data_dir,
    #     batch_size=1,
    #     image_size=32,
    #     class_cond=False,
    # )
    # image, cond = next(data)

    # save_image_to_neptune(run, image, "original")

    const_noise = th.randn((1,3,32,32)).to(dist_util.dev())

    # import types
    # diffusion.p_sample = types.MethodType(p_sample_const, diffusion)

    for i in range(10):
        # t = th.tensor([999]).to(dist_util.dev())
        # image = image.to(dist_util.dev())
        # logger.log(image.shape)
        # noised_image = diffusion.q_sample(image, t, CONST_NOISE)
        # logger.log(noised_image.shape)
        save_image_to_neptune(run, const_noise, f"noise {i}th time")
        unnoised_image = diffusion.p_sample_loop(model, (1,3,32,32), noise=const_noise)
        # logger.log(unnoised_image.shape)
        save_image_to_neptune(run, unnoised_image, f"unnoised {i}th time")

    run.stop()


def save_image_to_neptune(run, img, label):
    img = ((img + 1) * 127.5).clamp(0, 255).to(th.uint8)
    img = img.permute(0, 2, 3, 1)
    img = img.contiguous()
    # logger.log(img.shape)

    run["image_series"].append(
        File.as_image(
                img[0].cpu() / 255
            ),  # You can upload arrays as images using the File.as_image() method
            name=label,
        )


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
