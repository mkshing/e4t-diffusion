import os
import argparse
import hashlib
from tqdm.auto import tqdm
import blobfile as bf
import numpy as np
from PIL import Image
import albumentations
from einops import rearrange
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from diffusers import StableUnCLIPImg2ImgPipeline, DPMSolverMultistepScheduler
from diffusers.utils import is_xformers_available
from accelerate import Accelerator
from diffusers.utils import check_min_version

check_min_version("0.15.0.dev0")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--image_variation_dir", type=str, default="image_variation", help="output directory for stable unclip")
    parser.add_argument("--num_images_per_image", type=int, default=3, help="number of images to generate per input image by stable unclip")
    parser.add_argument("--train_image_dataset", type=str, default=None, required=True,
                        help="A folder containing the training data.")
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args()



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



def main():
    accelerator = Accelerator()
    args = parse_args()
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16,
        scheduler=DPMSolverMultistepScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", subfolder="scheduler")
    )
    if is_xformers_available():
        pipe.enable_xformers_memory_efficient_attention()
    pipe.set_progress_bar_config(disable=True)
    pipe.to(accelerator.device)
    sample_dataset = E4TDataset(
        dataset_name=args.train_image_dataset,
        resolution=args.resolution,
    )
    sample_dataloader = DataLoader(sample_dataset, batch_size=1)
    sample_dataloader = accelerator.prepare(sample_dataloader)

    os.makedirs(args.image_variation_dir, exist_ok=True)
    with torch.autocast("cuda"), torch.inference_mode():
        for example in tqdm(sample_dataloader, desc="Reimagining",):
            images = example["pixel_values"]
            images_to_log = []
            # to pil
            x_samples = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            for x_sample in x_samples:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))
                images_to_log.append(img)
            images = pipe(
                image=images_to_log[0].convert("RGB"),
                num_images_per_prompt=args.num_images_per_image
            ).images
            for i, image in enumerate(images):
                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = os.path.join(args.image_variation_dir, f"{hash_image}.jpg")
                image.save(image_filename)
    print("DONE!")

if __name__ == '__main__':
    main()
