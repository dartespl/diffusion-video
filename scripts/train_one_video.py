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

    logger.log("creating data loader...")
    data = get_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        deterministic=True,
        random_flip=False
    )
    import types
    video_frames = []
    logger.log("start")
    for iter, (X, cond) in enumerate(data):
        #  logger.log(iter, X.shape)
         video_frames.append(X)
    diffusion.video_frames = torch.cat(video_frames, dim=0)
    logger.log(diffusion.video_frames.shape)
    diffusion.q_sample = types.MethodType(q_sample_frame, diffusion)
    diffusion.q_posterior_mean_variance = types.MethodType(q_posterior_mean_variance, diffusion)
    diffusion._vb_terms_bpd = types.MethodType(_vb_terms_bpd, diffusion)

    logger.log("creating data loader...")
    loader = get_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        deterministic=False,
        random_flip=False,
        n_times=5
    )
    data = get_generator(loader)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        
        use_neptune=args.use_neptune
    ).run_loop()

def get_loader(
    *,
    data_dir,
    batch_size,
    image_size,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    n_times = 1
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    for _ in range(n_times):
         all_files += list(all_files)
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=None,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False
        )
    return loader
    
def get_generator(loader):
     while True:
        yield from loader


def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        logger.log("true_mean: ", true_mean)
        logger.log("true_log_variance_clipped: ", true_log_variance_clipped)
        logger.log("model mean: ", out["mean"])
        logger.log("model variance: ", out["log_variance"])
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}


def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape


        op1 = []
        op2 = []
        for ti in t:
             op1.append(self.video_frames[ti])
             op2.append(self.video_frames[ti-1])
        out1 = torch.stack(op1, dim=0).to(dist_util.dev())
        out2 = torch.stack(op2, dim=0).to(dist_util.dev())
        # logger.log(out1, out2)
        posterior_mean = out1-out2
        # logger.log(posterior_mean)


        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        
        # logger.log(posterior_variance.shape, posterior_log_variance_clipped.shape)
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

def q_sample_frame(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        # logger.log(t)
        out = []
        for ti in t:
             out.append(self.video_frames[ti])
            #  logger.log(self.video_frames[ti].shape)
        out = torch.stack(out, dim=0).to(dist_util.dev())
        # logger.log(out.shape)
        return out


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
