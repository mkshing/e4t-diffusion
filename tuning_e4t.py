import os
import argparse
from packaging import version
import json
from tqdm.auto import tqdm
import itertools
import random

import numpy as np
from PIL import Image
import albumentations
import torch
from torch.nn import functional as F

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import CLIPTokenizer
from diffusers import DDPMScheduler, AutoencoderKL
from diffusers.utils import is_xformers_available, load_image
from diffusers.optimization import get_scheduler
from e4t.models.modeling_clip import CLIPTextModel
from e4t.utils import load_config_from_pretrained, load_e4t_unet, load_e4t_encoder, save_e4t_unet, save_e4t_encoder
from pretrain_e4t import templates, art_templates, face_templates


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # e4t configs
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--domain_embed_scale", type=float, default=0.1, help="scale of e4t encoder's embedding")
    parser.add_argument("--reg_lambda", type=float, default=1e-4, help="l2 regularization lambda")
    parser.add_argument("--train_image_path", type=str, default=None, required=True, help="a image path or url")
    parser.add_argument("--prompt_template", type=str, default=None, help="If None, take the template from pretrained args. ")
    # training
    parser.add_argument("--unfreeze_clip_vision", action="store_true", default=False, help="train clip image encoder as a part of e4t encoder")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--learning_rate", type=float, default=1.6e-5, help="learning rate",)
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--max_train_steps", type=int, default=15, help="Total number of training steps to perform. For face, 30,000. For cat, 60,000. For art, 100,000",)
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--checkpointing_steps", type=int, default=10000,
                        help="Save a checkpoint of the training state every X updates.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",)
    # general
    parser.add_argument("--report_to", type=str, default=None, choices=["tensorboard", "wandb"])
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
    return args


def make_transforms(size, random_crop=False):
    rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=3)
    if not random_crop:
        cropper = albumentations.CenterCrop(height=size, width=size)
    else:
        cropper = albumentations.RandomCrop(height=size, width=size)
    flip = albumentations.HorizontalFlip(p=0.5)
    return albumentations.Compose([rescaler, cropper, flip])


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
    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )
    
    # load pre-trained args
    pretrained_args = load_config_from_pretrained(args.pretrained_model_name_or_path)
    # load pre-trained model
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_args.pretrained_model_name_or_path, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    vae = AutoencoderKL.from_pretrained(pretrained_args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = load_e4t_unet(
        pretrained_model_name_or_path=pretrained_args.pretrained_model_name_or_path,
        # load weight offsets from pre-trained model
        ckpt_path=os.path.join(args.pretrained_model_name_or_path, "weight_offsets.pt"),
    )
    # encoder
    e4t_encoder = load_e4t_encoder(
        word_embedding_dim=text_encoder.config.hidden_size,
        block_out_channels=unet.config.block_out_channels,
        clip_model=pretrained_args.clip_model_name_or_path,
        freeze_clip_vision=not args.unfreeze_clip_vision,
        ckpt_path=args.pretrained_model_name_or_path
    )
    print(f"Loaded the pre-trained model from {args.pretrained_model_name_or_path}")
    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(pretrained_args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(f"The tokenizer already contains the token {pretrained_args.placeholder_token}. Please pass a different `placeholder_token` that is not already in the tokenizer.")
    placeholder_token_id = tokenizer.convert_tokens_to_ids(pretrained_args.placeholder_token)
    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # freeze
    vae.requires_grad_(False)
    if not args.train_text_encoder:
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
    # # weight offsets
    # for n, p in unet.named_parameters():
    #     if "wo" in n:
    #         optim_params += [p]
    optim_params += list(unet.parameters())
    if args.train_text_encoder:
        optim_params += list(text_encoder.parameters())
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
    processor = make_transforms(args.resolution, random_crop=True)
    pil_image = load_image(args.train_image_path)
    image = np.array(pil_image)
    image = processor(image=image)["image"]
    pil_image_to_save = Image.fromarray(image)
    image = (image / 127.5 - 1.0).astype(np.float32)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    # For mixed precision training we cast the unet and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, e4t_encoder, optimizer, lr_scheduler = accelerator.prepare(unet, text_encoder, e4t_encoder, optimizer, lr_scheduler)
    else:
        unet, e4t_encoder, optimizer, lr_scheduler = accelerator.prepare(unet, e4t_encoder, optimizer, lr_scheduler)

    # Move vae and unet to device and cast to weight_dtype
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("e4t", config=vars(args))
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    def save_weights(step):
        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            unet_model = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
            e4t_enc_model = accelerator.unwrap_model(e4t_encoder, keep_fp32_wrapper=True)
            save_dir = os.path.join(args.output_dir, f"{step}")
            os.makedirs(save_dir, exist_ok=True)
            args_to_save = args.__dict__
            args_to_save["pretrained_args"] = pretrained_args.__dict__["obj"]
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump(args_to_save, f, indent=2)
            # save entire unet
            torch.save(unet_model.state_dict(), os.path.join(save_dir, "unet.pt"))
            # save encoder
            save_e4t_encoder(e4t_enc_model, save_dir)
            # save text encoder
            if args.train_text_encoder:
                torch.save(accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True).state_dict(), os.path.join(save_dir, "text_encoder.pt"))
            # save image
            pil_image_to_save.save(os.path.join(save_dir, "domain.png"))
            print(f"[*] Weights saved at {save_dir}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    first_epoch = 0
    global_step = 0
    unet.train()
    e4t_encoder.train()
    if args.train_text_encoder:
        text_encoder.train()
    domain_class_token_id = tokenizer(pretrained_args.domain_class_token, add_special_tokens=False, return_tensors="pt").input_ids[0]
    assert domain_class_token_id.size(0) == 1

    if args.prompt_template is None:
        args.prompt_template = pretrained_args.prompt_template
    if args.prompt_template in ["normal", "face", "art"]:
        if args.prompt_template == "normal":
            prompt_templates = templates
        elif args.prompt_template == "face":
            prompt_templates = face_templates
        else:
            prompt_templates = art_templates
        print(f"Using the default {len(prompt_templates)} templates!")
    else:
        assert "{placeholder_token}" in args.prompt_template, "You must specify the location of placeholder token by '{placeholder_token}'"
        prompt_templates = [args.prompt_template]
    pixel_values = image.expand(args.train_batch_size, -1, -1, -1)
    # Convert images to latent space
    latents = vae.encode(pixel_values.to(dtype=weight_dtype).to(accelerator.device)).latent_dist.sample().detach()
    latents = latents * vae.config.scaling_factor
    for step in range(args.max_train_steps):
        with accelerator.accumulate(unet):
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # get class token embedding
            text_embedding = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True).get_input_embeddings() if args.train_text_encoder else text_encoder.get_input_embeddings()
            class_embed = text_embedding(domain_class_token_id.to(accelerator.device)).detach()
            input_ids_for_encoder = tokenizer(
                "", padding="max_length", truncation=True, max_length=tokenizer.model_max_length,
                return_tensors="pt"
            ).input_ids
            # Get the text embedding for e4t conditioning
            encoder_hidden_states_for_e4t = text_encoder(input_ids_for_encoder.to(accelerator.device))[0].to(dtype=weight_dtype).detach()

            # get prompt
            batch_templates = random.choices(prompt_templates, k=bsz)
            prompt = [prompt_template.format(placeholder_token=pretrained_args.placeholder_token) for prompt_template in batch_templates]            
            input_ids = tokenizer(
                prompt, padding="max_length", truncation=True, max_length=tokenizer.model_max_length,
                return_tensors="pt"
            ).input_ids
            # Get the text embedding
            inputs_embeds = text_embedding(input_ids.to(accelerator.device))
            placeholder_token_id_idxs = [i.index(placeholder_token_id) for i in input_ids.cpu().tolist()]            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states_for_e4t_forward = encoder_hidden_states_for_e4t.expand(bsz, -1, -1).to(dtype=weight_dtype)
            # Get the unet encoder outputs
            encoder_outputs = unet(noisy_latents, timesteps, encoder_hidden_states_for_e4t_forward, return_encoder_outputs=True)
            # Forward E4T encoder to get the embedding
            domain_embed = e4t_encoder(x=pixel_values.to(accelerator.device), unet_down_block_samples=encoder_outputs["down_block_samples"])
            # update word embedding
            domain_embed = class_embed.clone().expand(bsz, -1) + args.domain_embed_scale * domain_embed

            for i, placeholder_token_id_idx in enumerate(placeholder_token_id_idxs):
                inputs_embeds[i, placeholder_token_id_idx, :] = domain_embed[i]

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(inputs_embeds=inputs_embeds)[0].to(dtype=weight_dtype)
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
                params_to_clip = (
                    itertools.chain(unet.parameters(), e4t_encoder.parameters(), text_encoder.parameters()) 
                    if args.train_text_encoder
                    else itertools.chain(unet.parameters(), e4t_encoder.parameters())
                )
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # # Checks if the accelerator has performed an optimization step behind the scenes
        # if accelerator.sync_gradients:
        progress_bar.update(1)
        global_step += 1
        if global_step % args.checkpointing_steps == 0:
            # if accelerator.is_main_process:
            #     save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            #     accelerator.save_state(save_path)
            #     print(f"Saved state to {save_path}")
            save_weights(global_step)

        logs = {
            "loss": loss.detach().item(),
            "loss_diff": loss_diff.detach().item(),
            "loss_reg": loss_reg.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0]
        }
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
    accelerator.wait_for_everyone()
    save_weights(global_step)
    accelerator.end_training()


if __name__ == '__main__':
    main()
