import argparse
from PIL import Image
import torch
from diffusers import (
    LMSDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)
from diffusers.utils import is_xformers_available
from e4t.models.modeling_clip import CLIPTextModel
from e4t.utils import load_config_from_pretrained, load_e4t_encoder, load_e4t_unet
from e4t.utils import load_image, AttributeDict
from e4t.pipeline_stable_diffusion_e4t import StableDiffusionE4TPipeline


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path_or_url", type=str, help="path to the input image")
    parser.add_argument("--pretrained_model_name_or_path", type=str, help="model dir including config.json, encoder.pt, weight_offsets.pt")
    # diffusers config
    parser.add_argument("--prompt", type=str, nargs="?", default="a photo of *s", help="the prompt to render")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="number of ddim sampling steps")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="unconditional guidance scale")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="number of images per prompt")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space",)
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space",)
    parser.add_argument("--seed", type=int, default=None, help="the seed (for reproducible sampling)")
    parser.add_argument("--scheduler_type", type=str, choices=["ddim", "plms", "lms", "euler", "euler_ancestral", "dpm_solver++"], default="ddim", help="diffusion scheduler type")
    opt = parser.parse_args()
    return opt


def getattr_from_config(config, key):
    if config.pretrained_args is not None:
        return config.pretrained_args[key]
    else:
        # pre-training phase model
        value = getattr(config, key)
        assert value is not None
        return value


def get_e4t_config(config):
    return AttributeDict(config.pretrained_args) if config.pretrained_args is not None else config


SCHEDULER_MAPPING = {
    "ddim": DDIMScheduler,
    "plms": PNDMScheduler,
    "lms": LMSDiscreteScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "dpm_solver++": DPMSolverMultistepScheduler,
}


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    # load models
    config = load_config_from_pretrained(args.pretrained_model_name_or_path)
    pretrained_model_name_or_path = getattr_from_config(config, "pretrained_model_name_or_path")
    unet = load_e4t_unet(
        ckpt_path=args.pretrained_model_name_or_path,
    )
    
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    e4t_encoder = load_e4t_encoder(
        ckpt_path=args.pretrained_model_name_or_path,
        word_embedding_dim=text_encoder.config.hidden_size,
        clip_model=getattr_from_config(config, "clip_model_name_or_path")
    )
    # load pipe
    pipe = StableDiffusionE4TPipeline.from_pretrained(
        pretrained_model_name_or_path,
        unet=unet,
        text_encoder=text_encoder,
        e4t_encoder=e4t_encoder,
        e4t_config=get_e4t_config(config),
        scheduler=SCHEDULER_MAPPING[args.scheduler_type].from_pretrained(pretrained_model_name_or_path, subfolder="scheduler"),
        requires_safety_checker=False,
        safety_checker=None,
        feature_extractor=None,
    )
    if is_xformers_available():
        pipe.enable_xformers_memory_efficient_attention()
        print("Using xformers!")
    pipe = pipe.to(device)
    print("loaded pipeline")
    # run!
    # download an image
    image = load_image(args.image_path_or_url)
    generator = None
    if args.seed:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.autocast(device), torch.inference_mode():
        images = pipe(
            args.prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            image=image,
            num_images_per_prompt=args.num_images_per_prompt,
            height=args.height,
            width=args.width,
        ).images
    grid_image = image_grid(images, 1, args.num_images_per_prompt)
    grid_image.save("grid.png")


if __name__ == '__main__':
    main()

