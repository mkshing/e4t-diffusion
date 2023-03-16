import os
import json
from requests.exceptions import HTTPError
import huggingface_hub

import numpy as np
import albumentations
from PIL import Image
import torch
from diffusers.utils import load_image as load_image_diffusers
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from e4t.models.unet_2d_condition import UNet2DConditionModel
from e4t.encoder import E4TEncoder


class AttributeDict(object):
    def __init__(self, obj):
        self.obj = obj

    def __getstate__(self):
        return self.obj.items()

    def __setstate__(self, items):
        if not hasattr(self, 'obj'):
            self.obj = {}
        for key, val in items:
            self.obj[key] = val

    def __getattr__(self, name):
        if name in self.obj:
            return self.obj.get(name)
        else:
            return None

    def fields(self):
        return self.obj

    def keys(self):
        return self.obj.keys()


def download_from_huggingface(repo, filename, **kwargs):
    while True:
        try:
            return huggingface_hub.hf_hub_download(
                repo,
                filename=filename,
                **kwargs
            )
        except HTTPError as e:
            if e.response.status_code == 401:
                # Need to log into huggingface api
                huggingface_hub.interpreter_login()
                continue
            elif e.response.status_code == 403:
                # Need to do the click through license thing
                print(
                    f"Go here and agree to the click through license on your account: https://huggingface.co/{repo}"
                )
                input("Hit enter when ready:")
                continue
            else:
                raise e


MODELS = {
    "mshing/e4t-diffusion/pretrained-something": {
        "repo": "mshing/e4t-diffusion",
        "subfolder": "pretrained-something"
    }
}
FILES = ["weight_offsets.pt", "encoder.pt", "config.json"]


def load_config_from_pretrained(pretrained_model_name_or_path):
    if os.path.exists(pretrained_model_name_or_path):
        if "config.json" not in pretrained_model_name_or_path:
            pretrained_model_name_or_path = os.path.join(pretrained_model_name_or_path, "config.json")
    else:
        assert pretrained_model_name_or_path in MODELS, f"Choose from {list(MODELS.keys())}"
        pretrained_model_name_or_path = download_from_huggingface(
            repo=MODELS[pretrained_model_name_or_path]["repo"],
            filename="config.json",
            subfolder=MODELS[pretrained_model_name_or_path]["subfolder"]
        )
    with open(pretrained_model_name_or_path, "r", encoding="utf-8") as f:
        pretrained_args = AttributeDict(json.load(f))
    return pretrained_args


def load_e4t_unet(pretrained_model_name_or_path=None, ckpt_path=None):
    assert pretrained_model_name_or_path is not None or ckpt_path is not None
    if pretrained_model_name_or_path is None:
        if os.path.exists(ckpt_path):
            if "weight_offsets.pt" in ckpt_path:
                ckpt_path = os.path.dirname(ckpt_path)
            config = load_config_from_pretrained(ckpt_path)
        else:
            assert ckpt_path in MODELS, f"Choose from {list(MODELS.keys())}"
            config = load_config_from_pretrained(ckpt_path)
            ckpt_path = download_from_huggingface(
                repo=MODELS[ckpt_path]["repo"],
                filename="weight_offsets.pt",
                subfolder=MODELS[ckpt_path]["subfolder"]
            )
        pretrained_model_name_or_path = config.pretrained_model_name_or_path if not hasattr(config, "pretrained_args") else config.pretrained_args["pretrained_model_name_or_path"]
    unet = OriginalUNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    state_dict = dict(unet.state_dict())
    if ckpt_path:
        if "weight_offsets.pt" not in ckpt_path:
            ckpt_path = os.path.join(ckpt_path, "weight_offsets.pt")
        wo_sd = torch.load(ckpt_path, map_location="cpu")
        state_dict.update(wo_sd)
        print(f"Resuming from {ckpt_path}")
    unet = UNet2DConditionModel(**unet.config)
    m, u = unet.load_state_dict(state_dict, strict=False)
    if len(m) > 0 and ckpt_path:
        raise RuntimeError(f"missing keys:\n{m}")
    if len(u) > 0:
        raise RuntimeError(f"unexpected keys:\n{u}")
    return unet


def save_e4t_unet(model, save_dir):
    weight_offsets_sd = {k: v for k, v in model.state_dict().items() if "wo" in k}
    torch.save(weight_offsets_sd, os.path.join(save_dir, "weight_offsets.pt"))


def load_e4t_encoder(ckpt_path=None, **kwargs):
    encoder = E4TEncoder(**kwargs)
    if ckpt_path:
        if os.path.exists(ckpt_path):
            if "encoder.pt" not in ckpt_path:
                ckpt_path = os.path.join(ckpt_path, "encoder.pt")
        else:
            assert ckpt_path in MODELS, f"Choose from {list(MODELS.keys())}"
            ckpt_path = download_from_huggingface(
                repo=MODELS[ckpt_path]["repo"],
                filename="encoder.pt",
                subfolder=MODELS[ckpt_path]["subfolder"]
            )
        state_dict = torch.load(ckpt_path, map_location="cpu")
        print(f"Resuming from {ckpt_path}")
        m, u = encoder.load_state_dict(state_dict, strict=False)
        m = [k for k in m if "clip_vision" not in k]
        if len(m) > 0:
            raise RuntimeError(f"missing keys:\n{m}")
        if len(u) > 0:
            raise RuntimeError(f"unexpected keys:\n{u}")

    return encoder


def save_e4t_encoder(model, save_dir):
    encoder_sd = {k: v for k, v in model.state_dict().items() if "clip_vision" not in k}
    torch.save(encoder_sd, os.path.join(save_dir, "encoder.pt"))


def make_transforms(size, random_crop=False):
    rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=3)
    if not random_crop:
        cropper = albumentations.CenterCrop(height=size, width=size)
    else:
        cropper = albumentations.RandomCrop(height=size, width=size)
    return albumentations.Compose([rescaler, cropper])


def load_image(image_path, resolution=None):
    pil_image = load_image_diffusers(image_path)
    if resolution:
        processor = make_transforms(resolution)
        image = np.array(pil_image)
        image = processor(image=image)["image"]
        pil_image = Image.fromarray(image)
    return pil_image
