"""Cross-modality translation for the MGnifyMultiOmics task (taxonomy -> KO).

Based on upstream `script/training_diffusion/py_scripts/multimodal_translation.py`
(commit d60d928) with exactly two changes, both required for this project's data:

  1. `n_cat` is read from the data instead of being hardcoded to 36 upstream
     (`n_cat=36,#label_encoder.classes_.shape[0]`). It is inert when
     covariate_specific_theta is False, but a correct value avoids surprises.

  2. The condition label mapping is taken from the *training* split
     (`biome_map.json`, written when the dataset was built) instead of refitting a
     LabelEncoder on the evaluation file. Upstream refits on the evaluation file,
     which silently permutes class indices whenever the evaluation split contains
     biome categories the train split does not (93 train classes here, several
     evaluation-only). Evaluation-only biomes are mapped to `fallback_index`
     (documented).

Everything else (sampling loop, DPM/ddpm branch, output .npz layout) is unchanged.
NOTE: use --sample_fn ddpm|ddim.  `DPM_Solver.sample()` does not accept `model_kwargs`,
so with --sample_fn dpm_solver* the conditioning is silently dropped and the model
samples unconditionally (upstream quirk).
"""
import sys
import os
import json
import argparse
import numpy as np
import torch as th
import torch.distributed as dist
from einops import rearrange, repeat
import muon as mu
import yaml

from scdiffusionX.DiffusionBackbone import dist_util, logger
from scdiffusionX.DiffusionBackbone.multimodal_script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from scdiffusionX.DiffusionBackbone.common import set_seed_logger_random, delete_pkl
from scdiffusionX.DiffusionBackbone.multimodal_dpm_solver_plus import DPM_Solver as multimodal_DPM_Solver
from scdiffusionX.DiffusionBackbone.dpm_solver_plus import DPM_Solver as singlemodal_DPM_Solver
from scdiffusionX.Autoencoder.models.base.encoder_model import EncoderModel


def main():
    args = create_argparser().parse_args()
    args.rna_dim = [int(i) for i in args.rna_dim.split(',')]
    args.atac_dim = [int(i) for i in args.atac_dim.split(',')]

    dist_util.setup_dist(args.devices)
    logger.configure(args.output_dir)
    args = set_seed_logger_random(args)

    logger.log("creating model and diffusion...")
    multimodal_model, multimodal_diffusion = create_model_and_diffusion(
        **args_to_dict(args, [key for key in model_and_diffusion_defaults().keys()])
    )

    if os.path.isdir(args.multimodal_model_path):
        multimodal_name_list = [model_name for model_name in os.listdir(args.multimodal_model_path)
                                if (model_name.startswith('model') and model_name.endswith('.pt')
                                    and int(model_name.split('.')[0][5:]) >= args.skip_steps)]
        multimodal_name_list.sort()
        multimodal_name_list = [os.path.join(args.model_path, m) for m in multimodal_name_list[::1]]
    else:
        multimodal_name_list = [p for p in args.multimodal_model_path.split(',')]

    logger.log(f"models waiting to be evaluated:{multimodal_name_list}")

    sr_noise = None
    if os.path.exists(args.load_noise):
        sr_noise = np.load(args.load_noise)
        sr_noise = th.tensor(sr_noise).to(dist_util.dev()).unsqueeze(0)
        sr_noise = repeat(sr_noise, 'b c h w -> (b repeat) c h w',
                          repeat=args.batch_size * args.rna_dim[0])
        if dist.get_rank() == 0:
            logger.log(f"load noise form {args.load_noise}...")

    for model_path in multimodal_name_list:
        multimodal_model.load_state_dict_(
            dist_util.load_state_dict(model_path, map_location="cpu"), is_strict=args.is_strict
        )
        multimodal_model.to(dist_util.dev())
        if args.use_fp16:
            multimodal_model.convert_to_fp16()
        multimodal_model.eval()

        logger.log(f"sampling samples for {model_path}")
        model_name = model_path.split('/')[-1]

        groups = 0
        multimodal_save_path = os.path.join(args.output_dir, model_name, 'original')
        audio_save_path = os.path.join(args.output_dir)
        img_save_path = os.path.join(args.output_dir)
        if dist.get_rank() == 0:
            os.makedirs(multimodal_save_path, exist_ok=True)
            os.makedirs(audio_save_path, exist_ok=True)
            os.makedirs(img_save_path, exist_ok=True)

        mdata = mu.read_h5mu(args.data_dir)
        if args.class_cond:
            # ---- CHANGE 2: use the training-time biome mapping, not a refit on evaluation data ----
            labels = mdata['rna'].obs[args.condition].astype(str).values
            with open(args.biome_map) as fh:
                bm = json.load(fh)
            bmap, fallback = bm["map"], bm.get("fallback_index", 0)
            n_unseen = int(sum(1 for l in labels if l not in bmap))
            classes_all = np.array([bmap.get(l, fallback) for l in labels], dtype=np.int64)
            if dist.get_rank() == 0:
                logger.log(f"condition '{args.condition}': {len(bmap)} train classes, "
                           f"{n_unseen}/{len(labels)} samples fall back to index {fallback}")
        else:
            classes_all = np.zeros(mdata['rna'].shape[0], dtype=np.int64)

        with open(args.encoder_config, 'r') as file:
            autoencoder_args = yaml.safe_load(file)

        # ---- CHANGE 1: n_cat from the data, not hardcoded 36 ----
        n_cat = len(np.unique(classes_all)) if args.class_cond else 0
        encoder_model = EncoderModel(in_dim={'atac': mdata['atac'].shape[1],
                                             'rna': mdata['rna'].shape[1]},
                                     n_cat=n_cat,
                                     conditioning_covariate=args.condition,
                                     encoder_type='learnt_autoencoder',
                                     **autoencoder_args)
        encoder_model.load_state_dict(th.load(args.ae_path)["state_dict"])
        encoder_model.to(dist_util.dev())

        batch = {}
        gt_rna = th.tensor(mdata['rna'].X.toarray(), device=dist_util.dev())
        gt_atac = th.tensor(mdata['atac'].X.toarray(), device=dist_util.dev())
        batch["X_norm"] = {'rna': gt_rna, 'atac': gt_atac}
        z = encoder_model.encode(batch)
        noise_init = z[next(s for s in z.keys() if s != args.gen_mode)]

        npzfile = np.load('/'.join(args.ae_path.split('/')[:-2]) + '/norm_factor.npz')
        std = npzfile['rna_std'] if args.gen_mode == 'atac' else npzfile['atac_std']
        noise_init = noise_init / th.tensor(std, device=noise_init.device)

        videos = []
        audios = []
        all_labels = []

        sample_num = noise_init.shape[0]
        num_iteration = int(sample_num / args.batch_size) + 1
        for i in list(range(num_iteration)) * args.gen_times:
            model_kwargs = {}
            x_T_init = noise_init[i * args.batch_size:(i + 1) * args.batch_size]
            if x_T_init.shape[0] == 0:
                continue
            if args.gen_mode == 'rna':
                model_kwargs["audio"] = x_T_init.unsqueeze(1).to(dist_util.dev())
            else:
                model_kwargs["video"] = x_T_init.unsqueeze(1).to(dist_util.dev())
            if args.class_cond:
                classes = classes_all[i * args.batch_size:(i + 1) * args.batch_size]
                classes = th.tensor(classes, device=dist_util.dev(), dtype=th.int)
                model_kwargs["label"] = classes

            shape = {"video": (args.batch_size if i != num_iteration - 1 else x_T_init.shape[0], *args.rna_dim),
                     "audio": (args.batch_size if i != num_iteration - 1 else x_T_init.shape[0], *args.atac_dim)}

            if args.sample_fn == 'dpm_solver':
                dpm_solver = multimodal_DPM_Solver(
                    model=multimodal_model,
                    alphas_cumprod=th.tensor(multimodal_diffusion.alphas_cumprod, dtype=th.float32))
                x_T = {"video": th.randn(shape["video"]).to(dist_util.dev()),
                       "audio": th.randn(shape["audio"]).to(dist_util.dev())}
                sample = dpm_solver.sample(x_T, steps=20, order=3, skip_type="logSNR", method="singlestep")
            elif args.sample_fn == 'dpm_solver++':
                dpm_solver = multimodal_DPM_Solver(
                    model=multimodal_model,
                    alphas_cumprod=th.tensor(multimodal_diffusion.alphas_cumprod, dtype=th.float32),
                    predict_x0=True, thresholding=True)
                x_T = {"video": th.randn(shape["video"]).to(dist_util.dev()),
                       "audio": th.randn(shape["audio"]).to(dist_util.dev())}
                sample = dpm_solver.sample(x_T, steps=20, order=2, skip_type="logSNR", method="adaptive")
            else:
                sample_fn = (multimodal_diffusion.conditional_p_sample_loop
                             if args.sample_fn == "ddpm" else multimodal_diffusion.ddim_sample_loop)
                sample = sample_fn(
                    multimodal_model,
                    shape=shape,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    use_fp16=args.use_fp16,
                    class_scale=args.classifier_scale,
                )

            video = sample["video"]
            audio = sample["audio"]
            all_videos = video.detach().cpu().numpy()
            all_audios = audio.detach().cpu().numpy()
            if args.class_cond:
                all_labels.append(classes.cpu().numpy())
            videos.append(all_videos)
            audios.append(all_audios)
            groups += 1
            dist.barrier()

        videos = np.concatenate(videos)
        audios = np.concatenate(audios)
        all_labels = np.concatenate(all_labels) if all_labels != [] else np.zeros(audios.shape[0])
        # keep only the first sample_num rows (gen_times>1 repeats the pass)
        videos, audios, all_labels = videos[:sample_num], audios[:sample_num], all_labels[:sample_num]

        np.savez(os.path.join(img_save_path, f"RNA_{dist.get_rank()}.npz"), data=videos, label=all_labels)
        np.savez(os.path.join(audio_save_path, f"ATAC_{dist.get_rank()}.npz"), data=audios, label=all_labels)

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        ref_path="",
        batch_size=16,
        sample_fn="ddpm",          # NOTE: ddpm/ddim required for translation (see module docstring)
        multimodal_model_path="",
        output_dir="",
        classifier_scale=0.0,
        devices=None,
        is_strict=True,
        all_save_num=1024,
        seed=42,
        load_noise="",
        data_dir="",
        gen_mode='rna',
        class_cond=True,
        encoder_config='encoder_multimodal',
        condition='biome',
        gen_times=1,
        ae_path='',
        biome_map='',
        skip_steps=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
