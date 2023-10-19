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
    # schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=200,
        image_size=32,
        class_cond=False,
        deterministic=True
    )
    image, cond = next(data)
    image = image[199:200]
    logger.log(image.shape)


    ###############
    model_kwargs = None
    shape = (1,3,32,32)
    progress = False

    device = dist_util.dev()
    assert isinstance(shape, (tuple, list))
    # if noise is not None:
    #     img = noise
    # else:
    #     img = th.randn(*shape, device=device)
    img = image.to(device)
    # img = th.randn(*shape, device=device)
    indices = list(range(diffusion.num_timesteps))[::-1]


    logger.log(img.shape, shape)

    if progress:
        # Lazy import so that we don't depend on tqdm.
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        save_image_to_neptune(run, img, f"image step {i}")
        t = th.tensor([i] * shape[0], device=device)
        with th.no_grad():
            model_output = model(img, diffusion._scale_timesteps(t))
            logger.log("output", model_output)
            logger.log("image", img)
            # model_output = th.permute(model_output, (0, 3, 1, 2))
            # logger.log(model_output.shape)
            # img = model_output
            img += model_output
    ################

    run.stop()


def save_image_to_neptune(run, img, label):
    img = ((img + 1) * 127.5).clamp(0, 255).to(th.uint8)
    img = img.permute(0, 2, 3, 1)
    img = img.contiguous()
    logger.log(img.shape)

    run["image_series"].append(
        File.as_image(
                img[0].cpu() / 255
            ),  # You can upload arrays as images using the File.as_image() method
            name=label,
        )

# def p_sample_from_partially_noised_loop(
#         diffusion,
#         model,
#         t,
#         shape,
#         noise=None,
#         clip_denoised=True,
#         denoised_fn=None,
#         cond_fn=None,
#         model_kwargs=None,
#         device=None,
#         progress=False,
#     ):
#         final = None
#         for sample in p_sample_from_partially_noised_loop_progressive(
#             diffusion,
#             model,
#             t,
#             shape,
#             noise=noise,
#             clip_denoised=clip_denoised,
#             denoised_fn=denoised_fn,
#             cond_fn=cond_fn,
#             model_kwargs=model_kwargs,
#             device=device,
#             progress=progress,
#         ):
#             final = sample
#         return final["sample"]


# def p_sample_from_partially_noised_loop_progressive(
#         diffusion,
#         model,
#         t,
#         shape,
#         noise=None,
#         clip_denoised=True,
#         denoised_fn=None,
#         cond_fn=None,
#         model_kwargs=None,
#         device=None,
#         progress=False,
#     ):
#         if device is None:
#             device = next(model.parameters()).device
#         assert isinstance(shape, (tuple, list))
#         # if noise is not None:
#         #     img = noise
#         # else:
#         #     img = th.randn(*shape, device=device)
#         img = diffusion.video_frames[0]
#         indices = list(range(diffusion.num_timesteps))[::-1]


#         logger.log(img.shape, noise.shape)

#         if progress:
#             # Lazy import so that we don't depend on tqdm.
#             from tqdm.auto import tqdm

#             indices = tqdm(indices)

#         for i in indices:
#             t = th.tensor([i] * shape[0], device=device)
#             with th.no_grad():
#                 model_output = model(img, diffusion._scale_timesteps(t), **model_kwargs)
#                 img += model_output


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
