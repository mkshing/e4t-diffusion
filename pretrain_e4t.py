import os
import argparse
import random

from packaging import version
import math
import json
from tqdm.auto import tqdm
import blobfile as bf
import itertools

import numpy as np
from PIL import Image
import albumentations
from einops import rearrange
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from transformers import CLIPTokenizer
from diffusers import DDPMScheduler, AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from e4t.models.modeling_clip import CLIPTextModel
from e4t.encoder import E4TEncoder
from e4t.pipeline_stable_diffusion_e4t import StableDiffusionE4TPipeline
from e4t.utils import load_e4t_unet, load_e4t_encoder, save_e4t_unet, save_e4t_encoder, image_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # e4t configs
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5", required=False, help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--clip_model_name_or_path", type=str, default="openai/clip-vit-large-patch14", required=False, help="clip's vision encoder model")
    parser.add_argument("--placeholder_token", type=str, default="*s", help="A token to use as a placeholder for the concept.",)
    parser.add_argument("--domain_class_token", type=str, default=None, required=True, help="Coarse-class token such as `face`, `cat`, pr `art`")
    parser.add_argument("--domain_embed_scale", type=float, default=0.1, help="scale of e4t encoder's embedding")
    parser.add_argument("--reg_lambda", type=float, default=0.01, help="l2 regularization lambda")
    parser.add_argument("--prompt_template", type=str, default="a photo of {placeholder_token}", help="{placeholder_token} will be replaced to placeholder_token.")
    parser.add_argument("--train_image_dataset", type=str, default=None, required=True,
                        help="A folder containing the training data.")
    parser.add_argument("--iterable_dataset", action="store_true", default=False)
    # training
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--learning_rate", type=float, default=1.6e-5, help="learning rate",)
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=1,)
    parser.add_argument("--max_train_steps", type=int, default=30000, help="Total number of training steps to perform. For face, 30,000. For cat, 60,000. For art, 100,000",)
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--checkpointing_steps", type=int, default=10000, help="Save a checkpoint of the training state every X updates.")
    parser.add_argument("--log_steps", type=int, default=1000, help="sample images ")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    # log
    parser.add_argument("--save_sample_prompt", type=str, default="a photo of *s,a photo of *s in the style of monet", help="split with ',' for multiple prompts")
    parser.add_argument("--n_save_sample", type=int, default=2, help="The number of samples per prompt")
    parser.add_argument("--save_guidance_scale", type=float, default=7.5, help="CFG for save sample.")
    parser.add_argument("--save_inference_steps", type=int, default=50, help="The number of inference steps for save sample.",)
    # general
    parser.add_argument("--report_to", type=str, default="wandb", choices=["tensorboard", "wandb"])
    parser.add_argument("--revision", type=str, default=None, required=False, help="Revision of pretrained model identifier from huggingface.co/models.", )
    parser.add_argument("--output_dir", type=str, default="e4t-model", help="The output directory where the model predictions and checkpoints will be written.", )
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_image_dataset is None:
        raise ValueError("You must specify a train data directory.")
    if args.domain_class_token is None:
        raise ValueError("You must specify a coarse-class token.")
    return args


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def make_transforms(size, random_crop=False):
    rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=3)
    if not random_crop:
        cropper = albumentations.CenterCrop(height=size, width=size)
    else:
        cropper = albumentations.RandomCrop(height=size, width=size)
    return albumentations.Compose([rescaler, cropper])


class E4TDataset(Dataset):
    def __init__(
            self,
            dataset_name,
            resolution=512,
    ):
        if os.path.isdir(dataset_name):
            self.dataset = _list_image_files_recursively(dataset_name)
        else:
            self.dataset = load_dataset(dataset_name, split="train")
        self.processor = make_transforms(resolution, random_crop=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = np.array(self.dataset[idx]["image"].convert("RGB"))
        image = self.processor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        return dict(
            pixel_values=image,
        )


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )
    if args.seed is not None:
        set_seed(args.seed)

    # load pre-trained model
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = load_e4t_unet(args.pretrained_model_name_or_path)
    # encoder
    e4t_encoder = load_e4t_encoder(
        word_embedding_dim=text_encoder.config.hidden_size,
        block_out_channels=unet.config.block_out_channels,
        clip_model=args.clip_model_name_or_path
    )

    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different `placeholder_token` that is not already in the tokenizer.")
    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)
    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # freeze
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if is_xformers_available() and args.enable_xformers_memory_efficient_attention:
        import xformers
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            print("[WARNING] xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.")
        unet.enable_xformers_memory_efficient_attention()
        print("Using xFormers!")
    # else:
    #     raise ValueError("xformers is not available. Make sure it is installed correctly")
    # Initialize the optimizer
    optim_params = [p for p in e4t_encoder.parameters() if p.requires_grad]
    # weight offsets
    for n, p in unet.named_parameters():
        if "wo" in n:
            optim_params += [p]
    total_params = sum(p.numel() for p in optim_params)
    print(f"Number of Trainable Parameters: {total_params * 1.e-6:.2f} M")

    if args.scale_lr:
        learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                learning_rate, args.gradient_accumulation_steps, accelerator.num_processes, args.train_batch_size, args.learning_rate))
        args.learning_rate = learning_rate
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        optim_params,
        lr=args.learning_rate,
    )

    # dataset
    if not args.iterable_dataset:
        train_dataset = E4TDataset(
            dataset_name=args.train_image_dataset,
            resolution=args.resolution,
        )
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    else:
        processor = make_transforms(args.resolution)

        def preprocess(example):
            image = np.array(example["image"])
            image = processor(image=image)["image"]
            image = (image / 127.5 - 1.0).astype(np.float32)
            image = torch.from_numpy(image).permute(2, 0, 1)
            return image

        def collate_fn(examples):
            image = torch.stack([preprocess(example) for example in examples])
            return dict(pixel_values=image)

        train_dataset = load_dataset(args.train_image_dataset, split="train", streaming=True)
        train_dataset = train_dataset.shuffle(seed=args.seed, buffer_size=10_000)
        train_dataset = train_dataset.with_format("torch")
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)

    images_to_log = []
    for batch in train_dataloader:
        images = batch["pixel_values"]
        # to pil
        x_samples = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        for x_sample in x_samples:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            images_to_log.append(img)
        if len(images_to_log) > 20:
            break
    # For mixed precision training we cast the unet and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # Move vae and unet to device and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    if not args.iterable_dataset:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, e4t_encoder, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        unet, e4t_encoder, optimizer, lr_scheduler, train_dataloader
    )

    if not args.iterable_dataset:
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    else:
        args.num_train_epochs = 10000000000000000000
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("e4t", config=vars(args))
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    print("***** Running training *****")
    if not args.iterable_dataset:
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    def txt2img(step):
        if accelerator.is_main_process:
            pipeline = StableDiffusionE4TPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet, keep_fp32_wrapper=True),
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                vae=vae,
                e4t_encoder=accelerator.unwrap_model(e4t_encoder, keep_fp32_wrapper=True),
                e4t_config=args,
                already_added_placeholder_token=True,
                safety_checker=None,
                requires_safety_checker=False,
                torch_dtype=torch.float16,
            )
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            if is_xformers_available():
                pipeline.enable_xformers_memory_efficient_attention()
            pipeline = pipeline.to(accelerator.device)
            g_cuda = torch.Generator(device=accelerator.device).manual_seed(args.seed)
            pipeline.set_progress_bar_config(disable=True)
            sample_dir = os.path.join(args.output_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            prompts = args.save_sample_prompt.split(",")
            image_list = []
            selected_images_to_log = random.sample(images_to_log, len(prompts)*args.n_save_sample)
            idx = 0
            with torch.autocast("cuda"), torch.inference_mode():
                for save_prompt in tqdm(prompts, desc="Generating samples"):
                    for i in range(args.n_save_sample):
                        images = pipeline(
                            save_prompt,
                            guidance_scale=args.save_guidance_scale,
                            num_inference_steps=args.save_inference_steps,
                            generator=g_cuda,
                            image=selected_images_to_log[idx],
                        ).images
                        image_list.append(images[0])
                        idx += 1
            input_grid = image_grid(selected_images_to_log, rows=len(prompts), cols=args.n_save_sample)
            sample_grid = image_grid(image_list, rows=len(prompts), cols=args.n_save_sample)
            input_grid.save(os.path.join(sample_dir, f"input-{step}.png"))
            sample_grid.save(os.path.join(sample_dir, f"sample-{step}.png"))
            if args.report_to == "wandb":
                import wandb
                accelerator.log(
                    {
                        "train/inputs": wandb.Image(input_grid),
                        "train/samples": wandb.Image(sample_grid)
                    },
                    step=step
                )
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def save_weights(step):
        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            unet_model = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
            e4t_enc_model = accelerator.unwrap_model(e4t_encoder, keep_fp32_wrapper=True)
            save_dir = os.path.join(args.output_dir, f"{step}")
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump(args.__dict__, f, indent=2)
            # save weight offsets
            save_e4t_unet(unet_model, save_dir)
            # save encoder
            save_e4t_encoder(e4t_enc_model, save_dir)
            print(f"[*] Weights saved at {save_dir}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    first_epoch = 0
    global_step = 0
    # save class embed
    domain_class_token_id = tokenizer(args.domain_class_token, add_special_tokens=False, return_tensors="pt").input_ids[0]
    assert domain_class_token_id.size(0) == 1
    # get class token embedding
    class_embed = text_encoder.get_input_embeddings()(domain_class_token_id.to(accelerator.device))
    # TODO: empty string is good for encoder?
    input_ids_for_encoder = tokenizer(
        # "",
        args.prompt_template.format(placeholder_token=args.domain_class_token),
        padding="max_length", truncation=True, max_length=tokenizer.model_max_length,
        return_tensors="pt"
    ).input_ids
    prompt = args.prompt_template.format(placeholder_token=args.placeholder_token)
    print(f"prompt: {prompt}")
    input_ids = tokenizer(
        prompt, padding="max_length", truncation=True, max_length=tokenizer.model_max_length,
        return_tensors="pt"
    ).input_ids
    placeholder_token_id_idx = input_ids[0].tolist().index(placeholder_token_id)
    # Get the text embedding for e4t conditioning
    encoder_hidden_states_for_e4t = text_encoder(input_ids_for_encoder.to(accelerator.device))[0].to(dtype=weight_dtype)
    # Get the text embedding
    inputs_embeds = text_encoder.get_input_embeddings()(input_ids.to(accelerator.device))
    try:
        for epoch in range(first_epoch, args.num_train_epochs):
            unet.train()
            e4t_encoder.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                    latents = vae.encode(pixel_values).latent_dist.sample().detach()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    encoder_hidden_states_for_e4t_forward = encoder_hidden_states_for_e4t.expand(bsz, -1, -1)
                    # Get the unet encoder outputs
                    encoder_outputs = unet(noisy_latents, timesteps, encoder_hidden_states_for_e4t_forward, return_encoder_outputs=True)
                    # Forward E4T encoder to get the embedding
                    domain_embed = e4t_encoder(x=pixel_values, unet_down_block_samples=encoder_outputs["down_block_samples"])
                    # update word embedding
                    domain_embed = class_embed.clone().expand(bsz, -1) + args.domain_embed_scale * domain_embed
                    inputs_embeds_forward = inputs_embeds.expand(bsz, -1, -1).clone()
                    inputs_embeds_forward[:, placeholder_token_id_idx, :] = domain_embed
                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(inputs_embeds=inputs_embeds_forward)[0].to(dtype=weight_dtype)
                    # Predict the noise residual
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    # compute loss
                    loss_diff = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    loss_reg = args.reg_lambda * domain_embed.pow(2).sum()
                    loss = loss_diff + loss_reg
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = itertools.chain(unet.parameters(), e4t_encoder.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if global_step % args.checkpointing_steps == 0:
                        # save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # accelerator.save_state(save_path)
                        # print(f"Saved state to {save_path}")
                        save_weights(global_step)
                    # log at first step
                    if global_step == 1 or global_step % args.log_steps == 0:
                        txt2img(global_step)

                logs = {
                    "train/loss": loss.detach().item(),
                    "train/loss_diff": loss_diff.detach().item(),
                    "train/loss_reg": loss_reg.detach().item(),
                    "train/lr": lr_scheduler.get_last_lr()[0]
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break
    except KeyboardInterrupt:
        print("Summoning checkpoint...")
        pass
    accelerator.wait_for_everyone()
    save_weights(global_step)
    accelerator.end_training()


if __name__ == '__main__':
    main()
