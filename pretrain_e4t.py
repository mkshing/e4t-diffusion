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
import braceexpand
import webdataset as wds
import wandb

from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from transformers import CLIPTokenizer
from diffusers import DDPMScheduler, AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from e4t.models.modeling_clip import CLIPTextModel
from e4t.encoder import E4TEncoder
from e4t.pipeline_stable_diffusion_e4t import StableDiffusionE4TPipeline
from e4t.utils import load_e4t_unet, load_e4t_encoder, save_e4t_unet, save_e4t_encoder, image_grid


templates = [
    "a photo of {placeholder_token}",
    "the photo of {placeholder_token}",
    "a photo of a {placeholder_token}",
    "a photo of the {placeholder_token}",
    "a photo of one {placeholder_token}",
    "a close-up photo of the {placeholder_token}",
    "a bright photo of the {placeholder_token}",
    "a photo of a nice {placeholder_token}",
    "a good photo of {placeholder_token}",
    "a photo of a cool {placeholder_token}"
]

face_templates = templates + [
    "a portrait of {placeholder_token}",
    "the portrait of {placeholder_token}",
    "a portrait photo of {placeholder_token}",
    "portrait of {placeholder_token}",
    "portrait of the {placeholder_token}",
    "photo realistic portrait of {placeholder_token}",
]

art_templates = templates + [
    "art of {placeholder_token}",
    "art by {placeholder_token}",
    # more!
]



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # e4t configs
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5", required=False, help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--clip_model_name_or_path", type=str, default="ViT-H-14::laion2b_s32b_b79k", required=False, help="load from open_clip with the format 'arch::version'")
    parser.add_argument("--placeholder_token", type=str, default="*s", help="A token to use as a placeholder for the concept.",)
    parser.add_argument("--domain_class_token", type=str, default=None, required=True, help="Coarse-class token such as `face`, `cat`, pr `art`")
    parser.add_argument("--domain_embed_scale", type=float, default=0.1, help="scale of e4t encoder's embedding")
    parser.add_argument("--reg_lambda", type=float, default=0.01, help="l2 regularization lambda")
    parser.add_argument("--prompt_template", type=str, default="a photo of {placeholder_token}", help="{placeholder_token} will be replaced to placeholder_token. If you choose from ['normal', 'face', 'art'],use default multiple templates")
    parser.add_argument("--train_image_dataset", type=str, default=None, required=True,
                        help="A folder containing the training data.")
    parser.add_argument("--unfreeze_clip_vision", action="store_true", default=False, help="train clip image encoder as a part of e4t encoder")
    parser.add_argument("--webdataset", action="store_true", default=False, help="load tar files via webdataset")
    parser.add_argument("--iterable_dataset", action="store_true", default=False, help="Use iterable dataset in datasets")
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
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help=("Whether training should be resumed from a previous checkpoint. Use a path saved by"
        ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
    ))
    parser.add_argument("--log_steps", type=int, default=1000, help="sample images ")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    # log
    parser.add_argument("--save_sample_prompt", type=str, default="a photo of *s,a photo of *s in the style of monet", help="split with ',' for multiple prompts")
    parser.add_argument("--n_save_sample", type=int, default=4, help="The number of samples per prompt")
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
    flip = albumentations.HorizontalFlip(p=0.5)
    return albumentations.Compose([rescaler, cropper, flip])


class E4TDataset(Dataset):
    def __init__(
            self,
            dataset_name,
            resolution=512,
    ):
        super().__init__()
        from_datasets = False
        if os.path.isdir(dataset_name) or "::" in dataset_name:
            self.dataset = []
            for name in dataset_name.split("::"):
                self.dataset += _list_image_files_recursively(name)
        else:
            self.dataset = load_dataset(dataset_name, split="train")
            from_datasets = True
        self.from_datasets = from_datasets
        self.processor = make_transforms(resolution, random_crop=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]
        if self.from_datasets:
            image = image["image"]
        else:
            image = Image.open(image)
        image = np.array(image.convert("RGB"))
        image = self.processor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        return dict(
            pixel_values=image,
        )


def get_dataset_size(shards):
    shards_list = []
    for s in shards.split("::"):
        shards_list += list(braceexpand.braceexpand(s))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        # if "total_size" in sizes.keys():
        #     total_size = sizes['total_size']
        # else:
        #     total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.join(shards_list[0].replace('.tar', "_stats.json")):
        total_size = 0
        for shard in shards_list:
            json_path = shard.replace('.tar', "_stats.json")
            if os.path.exists(json_path):
                sizes = json.load(open(json_path))
                if 'n_data' in sizes:
                    size = sizes['n_data']
                else:
                    size = sizes["successes"]
                total_size += int(size)
            else:
                print(f"Not Found {json_path}")
    else:
        total_size = None  # num samples undefined
    num_shards = len(shards_list)
    return total_size, num_shards


def filter_webdataset(example):
    if "jpg" not in example or example["jpg"] is None:
        return False
    return True


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
    unet = load_e4t_unet(
        args.pretrained_model_name_or_path, 
        ckpt_path=os.path.join(args.pretrained_model_name_or_path, "weight_offsets.pt") if os.path.exists(os.path.join(args.pretrained_model_name_or_path, "weight_offsets.pt")) else None,
        revision=args.revision
    )
    # encoder
    e4t_encoder = load_e4t_encoder(
        word_embedding_dim=text_encoder.config.hidden_size,
        block_out_channels=unet.config.block_out_channels,
        arch=args.clip_model_name_or_path.split("::")[0],
        version=args.clip_model_name_or_path.split("::")[1],
        freeze_clip_vision=not args.unfreeze_clip_vision,
        ckpt_path=os.path.join(args.pretrained_model_name_or_path, "encoder.pt") if os.path.exists(os.path.join(args.pretrained_model_name_or_path, "weight_offsets.pt")) else None,
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


    # dataset
    if not args.iterable_dataset and not args.webdataset:
        train_dataset = E4TDataset(
            dataset_name=args.train_image_dataset,
            resolution=args.resolution,
        )
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    else:
        processor = make_transforms(args.resolution, random_crop=True)

        def preprocess(example):
            image_key = "image" if not args.webdataset else "jpg"
            image = np.array(example[image_key]).astype(np.uint8)
            image = processor(image=image)["image"]
            image = (image / 127.5 - 1.0).astype(np.float32)
            image = torch.from_numpy(image).permute(2, 0, 1)
            return image

        def collate_fn(examples):
            image = torch.stack([preprocess(example) for example in examples])
            return dict(pixel_values=image)

        if args.webdataset:
            num_samples, num_shards = get_dataset_size(args.train_image_dataset)
            print(f'Loading webdataset with {num_shards} shards. (num_samples: {num_samples})')
            pipeline = [wds.ResampledShards(args.train_image_dataset)]
            pipeline.extend([
                wds.split_by_node,
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker at each node
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
                wds.shuffle(1000, handler=wds.warn_and_continue),
            ])
            pipeline.extend([
                wds.select(filter_webdataset),
                wds.decode("pilrgb", handler=wds.warn_and_continue),
                wds.map(preprocess, handler=wds.warn_and_continue),
                # wds.batched(args.train_batch_size, partial=False, collation_fn=lambda images: dict(pixel_values=torch.stack(images))),
            ])
            train_dataset = wds.DataPipeline(*pipeline)
            world_size = accelerator.num_processes
            assert num_shards >= args.dataloader_num_workers * world_size, 'number of shards must be >= total workers'
            global_batch_size = args.train_batch_size * world_size
            num_batches = math.ceil(num_samples / global_batch_size)
            num_worker_batches = math.ceil(num_batches / args.dataloader_num_workers)  # per dataloader worker
            num_batches = num_worker_batches * args.dataloader_num_workers
            num_samples = num_batches * global_batch_size
            train_dataset = train_dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
            train_dataloader = DataLoader(
                train_dataset,
                persistent_workers=True,
                drop_last=True,
                num_workers=args.dataloader_num_workers,
                collate_fn=lambda images: dict(pixel_values=torch.stack(images))
            )
            # train_dataloader = wds.WebLoader(
            #     train_dataset,
            #     batch_size=None,
            #     shuffle=False,
            #     num_workers=args.dataloader_num_workers,
            #     persistent_workers=True,
            # )
            # # add meta-data to dataloader instance for convenience
            # train_dataloader.num_batches = num_batches
            # train_dataloader.num_samples = num_samples
        else:
            train_dataset = load_dataset(args.train_image_dataset, split="train", streaming=True)
            train_dataset = train_dataset.shuffle(seed=args.seed, buffer_size=10000)
            train_dataset = train_dataset.with_format("torch")
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)

    if args.scale_lr:
        learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                learning_rate, args.gradient_accumulation_steps, accelerator.num_processes, args.train_batch_size, args.learning_rate))
        args.learning_rate = learning_rate

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if accelerator.unwrap_model(e4t_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(e4t_encoder).dtype}."
            f" {low_precision_error_string}"
        )
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

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    if not args.iterable_dataset and not args.webdataset:
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

    if not args.iterable_dataset and not args.webdataset:
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    else:
        args.num_train_epochs = 1000000000000000000000000000000000000
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("e4t", config=vars(args))
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    print("***** Running training *****")
    if not args.iterable_dataset and not args.webdataset:
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    @torch.no_grad()
    def sample(images, step):
        images_to_log = []
        # to pil
        x_samples = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        for x_sample in x_samples:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            images_to_log.append(img)
        pipeline = StableDiffusionE4TPipeline(
            unet=accelerator.unwrap_model(unet, keep_fp32_wrapper=True),
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            e4t_encoder=accelerator.unwrap_model(e4t_encoder, keep_fp32_wrapper=True),
            e4t_config=args,
            already_added_placeholder_token=True,
            requires_safety_checker=False,
            safety_checker=None,
            feature_extractor=None,
        )

        if is_xformers_available():
            pipeline.enable_xformers_memory_efficient_attention()
        pipeline = pipeline.to(accelerator.device)
        g_cuda = torch.Generator(device=accelerator.device)
        g_cuda = g_cuda.manual_seed(int(g_cuda.seed()))
        # g_cuda = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        pipeline.set_progress_bar_config(disable=True)
        sample_dir = os.path.join(args.output_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        prompts = args.save_sample_prompt.split(",")
        image_list = []
        selected_images_to_log = random.sample(images_to_log, min(len(images_to_log), args.n_save_sample))
        with torch.autocast("cuda"), torch.inference_mode():
            for save_prompt in tqdm(prompts, desc="Generating samples"):
                for image in selected_images_to_log:
                    images = pipeline(
                        save_prompt,
                        guidance_scale=args.save_guidance_scale,
                        num_inference_steps=args.save_inference_steps,
                        generator=g_cuda,
                        image=image,
                    ).images
                    image_list.append(images[0])
        input_grid = image_grid(selected_images_to_log, rows=1, cols=len(selected_images_to_log))
        sample_grid = image_grid(image_list, rows=len(prompts), cols=len(selected_images_to_log))
        if args.report_to == "wandb":
            accelerator.log(
                {
                    "train/inputs": wandb.Image(input_grid),
                    "train/samples": wandb.Image(sample_grid)
                },
                step=step
            )
        else:
            input_grid.save(os.path.join(sample_dir, f"input-{step}.png"))
            sample_grid.save(os.path.join(sample_dir, f"sample-{step}.png"))
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
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # save class embed
    domain_class_token_id = tokenizer(args.domain_class_token, add_special_tokens=False, return_tensors="pt").input_ids[0]
    assert domain_class_token_id.size(0) == 1
    # get class token embedding
    class_embed = text_encoder.get_input_embeddings()(domain_class_token_id.to(accelerator.device))
    input_ids_for_encoder = tokenizer(
        "",
        # args.prompt_template.format(placeholder_token=args.domain_class_token),
        padding="max_length", truncation=True, max_length=tokenizer.model_max_length,
        return_tensors="pt"
    ).input_ids
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
    # Get the text embedding for e4t conditioning
    encoder_hidden_states_for_e4t = text_encoder(input_ids_for_encoder.to(accelerator.device))[0].to(dtype=weight_dtype)

    try:
        for epoch in range(first_epoch, args.num_train_epochs):
            unet.train()
            e4t_encoder.train()
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    pixel_values = batch["pixel_values"]
                    latents = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample().detach()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()
                    
                    # get prompt
                    batch_templates = random.choices(prompt_templates, k=bsz)
                    prompt = [prompt_template.format(placeholder_token=args.placeholder_token) for prompt_template in batch_templates]
                    input_ids = tokenizer(
                        prompt, padding="max_length", truncation=True, max_length=tokenizer.model_max_length,
                        return_tensors="pt"
                    ).input_ids
                    # Get the text embedding
                    inputs_embeds = text_encoder.get_input_embeddings()(input_ids.to(accelerator.device))
                    placeholder_token_id_idxs = [i.index(placeholder_token_id) for i in input_ids.cpu().tolist()]

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
                    # if accelerator.sync_gradients:
                    #     params_to_clip = itertools.chain(unet.parameters(), e4t_encoder.parameters())
                    #     accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # # Checks if the accelerator has performed an optimization step behind the scenes
                # if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.checkpointing_steps == 0:
                    save_weights(global_step)
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    print(f"Saved state to {save_path}")
                # log at first step
                if global_step == 1 or global_step % args.log_steps == 0:
                    images = accelerator.gather(batch["pixel_values"])
                    if accelerator.is_main_process:
                        sample(images, global_step)

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
