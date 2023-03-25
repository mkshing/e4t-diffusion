import kornia

import torch
from torch import nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
import open_clip


class E4TEncoderLegacy(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            word_embedding_dim=768,
            block_out_channels=(320, 640, 1280, 1280),
            clip_model="openai/clip-vit-large-patch14",
            antialias=False,
            freeze_clip_vision=True,
    ):
        super().__init__()
        self.clip_vision = CLIPVisionModel.from_pretrained(clip_model)
        if freeze_clip_vision:
            self.clip_vision.requires_grad_(False)
        # encoder
        self.linear = nn.Linear(
            self.clip_vision.config.hidden_size,
            self.clip_vision.config.hidden_size,
        )
        self.act = nn.LeakyReLU()
        self.final_linear = nn.Linear(
            self.clip_vision.config.hidden_size+sum(block_out_channels),
            word_embedding_dim,
        )

        self.image_size = self.clip_vision.config.image_size
        self.antialias = antialias
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(
            x, (self.image_size, self.image_size), interpolation='bicubic', align_corners=True, antialias=self.antialias
        )
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x, unet_down_block_samples: tuple):
        """
        Inputs:
            - x: tensor of image
        """
        # clip feature extractor
        # x is assumed to be in range [-1,1]
        x = self.preprocess(x)
        outputs = self.clip_vision(pixel_values=x, output_hidden_states=True)
        # in huggingface implementation, the 1st hidden states represents the output of input embedding.
        clip_hidden_states = outputs.hidden_states[1:]
        clip_hidden_states = clip_hidden_states[1::2]  # take every 2nd layer
        # TODO: need normalization at last?
        clip_hidden_states = [
            self.linear(self.clip_vision.vision_model.post_layernorm(hidden_states[:, 0, :])) for hidden_states in
            clip_hidden_states
        ]
        clip_hidden_states = torch.stack(clip_hidden_states)
        clip_hidden_states = torch.mean(clip_hidden_states, dim=0)
        # unet pooling
        pooled_outputs = [self.act(sample.mean(dim=(2, 3))) for sample in unet_down_block_samples]
        pooled_outputs = [self.act(clip_hidden_states)] + pooled_outputs
        pooled_outputs = torch.cat(pooled_outputs, dim=1)
        # final linear layer
        return self.final_linear(pooled_outputs)


class E4TEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            word_embedding_dim=768,
            block_out_channels=(320, 640, 1280, 1280),
            arch="ViT-H-14", 
            version="laion2b_s32b_b79k",
            antialias=False,
            freeze_clip_vision=True,
            **kwargs
    ):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.transformer
        self.clip_vision = model.visual
        self.clip_vision.output_tokens = True
        # remove proj
        self.clip_vision.proj = None
        clip_vision_hidden_size = self.clip_vision.ln_post.normalized_shape[0]
        if freeze_clip_vision:
            self.clip_vision.requires_grad_(False)
        # unet
        self.unet_feature_embedder = nn.Sequential(
            nn.Linear(10880, clip_vision_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(clip_vision_hidden_size, clip_vision_hidden_size)
        )
        self.feature_linear = nn.Linear(2*clip_vision_hidden_size, clip_vision_hidden_size)

        self.first_linears = nn.ModuleList([])
        if arch == "ViT-H-14":
            # every odd resblock in the OpenCLIP implementation
            n_odd_layers = 128 + 1
        else:
            n_odd_layers = kwargs.get("n_odd_layers", None)
            assert n_odd_layers is not None, "You must specify `n_odd_layers`!"
            n_odd_layers = int(n_odd_layers)

        for _ in range(n_odd_layers):
            self.first_linears.append(
                nn.Linear(
                    clip_vision_hidden_size,
                    clip_vision_hidden_size,
                )
            )
        self.act = nn.LeakyReLU()
        self.final_linear = nn.Linear(clip_vision_hidden_size, word_embedding_dim)
        self.image_size = 224
        self.antialias = antialias
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(
            x, (self.image_size, self.image_size), interpolation='bicubic', align_corners=True, antialias=self.antialias
        )
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x, unet_down_block_samples: tuple):
        """
        Inputs:
            - x: tensor of image
        """
        # unet pooling
        unet_down_block_samples = [sample.mean(dim=(2, 3)) for sample in unet_down_block_samples]
        unet_pooled_features = torch.cat(unet_down_block_samples, dim=-1)
        unet_pooled_features = self.unet_feature_embedder(unet_pooled_features)

        # clip feature extractor
        # x is assumed to be in range [-1,1]
        x = self.preprocess(x)
        pooled_output, hidden_states = self.clip_vision(x)
        hidden_states = hidden_states[:, 1::2, :] # take every 2nd layer 
        hidden_states = torch.cat([pooled_output.unsqueeze(1), hidden_states], dim=1)
        n_layers = hidden_states.size(1)
        clip_hidden_states_list = []
        for i in range(n_layers):
            clip_hidden_states = self.feature_linear(torch.cat([hidden_states[:, i, :], unet_pooled_features], dim=-1))
            clip_hidden_states = self.first_linears[i](clip_hidden_states)
            clip_hidden_states_list.append(clip_hidden_states)
        clip_hidden_states = torch.stack(clip_hidden_states_list)
        # average pooling
        clip_hidden_states = torch.mean(clip_hidden_states, dim=0)
        clip_hidden_states = self.act(clip_hidden_states)
        # final linear layer
        return self.final_linear(clip_hidden_states)


if __name__ == '__main__':
    import numpy as np
    import skimage
    import albumentations

    from transformers import CLIPTokenizer
    from diffusers import AutoencoderKL, DDPMScheduler
    from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
    from e4t.models.unet_2d_condition import UNet2DConditionModel
    from e4t.models.modeling_clip import CLIPTextModel
    from e4t.weightoffsets import WeightOffsets

    def make_transforms(size, random_crop=False):
        rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=3)
        if not random_crop:
            cropper = albumentations.CenterCrop(height=size, width=size)
        else:
            cropper = albumentations.RandomCrop(height=size, width=size)
        return albumentations.Compose([rescaler, cropper])


    model_id = "CompVis/stable-diffusion-v1-4"
    img = skimage.data.astronaut() # numpy uint8
    resolution = 512
    placeholder_token = "*s"
    class_token = "face"
    text = "a photo of *s"
    domain_embed_scale = 0.1
    reg_lambda = 0.01
    # load model
    unet = OriginalUNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    state_dict = dict(unet.state_dict())
    # wo_sd = torch.load("weight_offsets.pt", map_location="cpu")
    # state_dict.update(wo_sd)
    unet = UNet2DConditionModel(**unet.config)
    m, u = unet.load_state_dict(state_dict, strict=False)
    # if len(m) > 0:
    #     raise RuntimeError(f"missing keys:\n{m}")
    if len(u) > 0:
        raise RuntimeError(f"unexpected keys:\n{u}")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    e4t_encoder = E4TEncoder(
        word_embedding_dim=text_encoder.config.hidden_size,
        block_out_channels=unet.config.block_out_channels,
        arch="ViT-H-14",
        version="laion2B-s34B-b79K"
    )
    processor = make_transforms(resolution)
    print("loaded models")
    # optimizer
    # encoder
    optim_params = [p for p in e4t_encoder.parameters() if p.requires_grad]
    # optim_params = []
    # weight offsets
    for n, p in unet.named_parameters():
        if "wo" in n:
            optim_params += [p]
    total_params = sum(p.numel() for p in optim_params)
    print(f"{e4t_encoder.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    optimizer = torch.optim.AdamW(optim_params, lr=1e-3)
    a = optim_params[-1].data.clone()
    b = optim_params[5].data.clone()
    
    # unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # add new token
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    text_encoder.resize_token_embeddings(len(tokenizer))
    # Prepare inputs
    img = processor(image=img)["image"]
    img = (img / 127.5 - 1.0).astype(np.float32)
    pixel_values = torch.from_numpy(img).permute(2, 0, 1)
    bsz = 3
    pixel_values = pixel_values.unsqueeze(0).expand(bsz, -1, -1, -1)
    latents = vae.encode(pixel_values).latent_dist.sample().detach()
    latents = latents * vae.config.scaling_factor
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(0, 1000, (bsz,), device=latents.device)
    timesteps = timesteps.long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    # encoder_hidden_states = torch.randn(bsz, 77, 768)
    # TODO: empty string is good for encoder?
    input_ids_for_e4t = tokenizer("", padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
    input_ids_for_e4t = input_ids_for_e4t.expand(bsz, -1)
    input_ids = tokenizer(text, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
    input_ids = input_ids.expand(bsz, -1)
    encoder_hidden_states_for_e4t = text_encoder(input_ids_for_e4t)[0]
    class_token_id = tokenizer(class_token, add_special_tokens=False, return_tensors="pt").input_ids[0]
    assert class_token_id.size(0) == 1
    # get class token embedding
    class_embed = text_encoder.get_input_embeddings()(class_token_id)

    # Train!
    unet.train()
    e4t_encoder.train()
    optimizer.zero_grad()

    # run encoder!
    encoder_outputs = unet(noisy_latents, timesteps, encoder_hidden_states_for_e4t, return_encoder_outputs=True)
    domain_embed = e4t_encoder(x=pixel_values, unet_down_block_samples=encoder_outputs["down_block_samples"])
    # update word embedding
    domain_embed = class_embed.clone().expand(bsz, -1) + domain_embed_scale * domain_embed

    token_embeds = text_encoder.get_input_embeddings().weight.data
    inputs_embeds_forward = token_embeds.expand(bsz, -1, -1).clone()
    inputs_embeds_forward[:, placeholder_token_id, :] = domain_embed

    encoder_hidden_states = text_encoder(input_ids)[0]
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    loss_diff = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
    loss_reg = reg_lambda * domain_embed.pow(2).sum()
    loss = loss_diff + loss_reg
    print(f"loss: {loss}, loss_diff: {loss_diff}, loss_reg: {loss_reg}")
    loss.backward()
    optimizer.step()
    print(torch.equal(a, optim_params[-1].data))
    print(torch.equal(b, optim_params[5].data))
    # weight_offsets_sd = {k: v for k, v in unet.state_dict().items() if "wo" in k}
    # torch.save(weight_offsets_sd, "weight_offsets.pt")
