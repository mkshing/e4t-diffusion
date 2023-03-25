"""
Originally taken from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
"""
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import PIL
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput


def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        image = [np.array(i)[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


class StableDiffusionE4TPipeline(StableDiffusionPipeline):
    def __init__(
            self,
            vae,
            text_encoder,
            tokenizer,
            unet,
            e4t_encoder,
            scheduler,
            safety_checker,
            feature_extractor,
            e4t_config = None,
            requires_safety_checker: bool = True,
            already_added_placeholder_token: bool = False,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker)
        # Add the placeholder token in tokenizer
        if not already_added_placeholder_token:
            num_added_tokens = self.tokenizer.add_tokens(e4t_config.placeholder_token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {e4t_config.placeholder_token}. Please pass a different `placeholder_token` that is not already in the tokenizer.")
            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            text_encoder.resize_token_embeddings(len(tokenizer))
        self.placeholder_token = e4t_config.placeholder_token
        self.placeholder_token_id = tokenizer.convert_tokens_to_ids(e4t_config.placeholder_token)
        # save class embed
        domain_class_token_id = self.tokenizer(e4t_config.domain_class_token, add_special_tokens=False, return_tensors="pt").input_ids[0]
        assert domain_class_token_id.size(0) == 1
        # get class token embedding
        self.class_embed = text_encoder.get_input_embeddings()(domain_class_token_id.to(text_encoder.device))
        self.domain_embed_scale = e4t_config.domain_embed_scale
        self.register_modules(e4t_encoder=e4t_encoder)

    def prepare_for_e4t(self, prompt, device):
        input_ids_for_encoder = self.tokenizer(
            "",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids
        input_ids = self.tokenizer(
            prompt, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids
        try:
            placeholder_token_id_idx = input_ids[0].tolist().index(self.placeholder_token_id)
        except ValueError:
            raise ValueError(f"Your prompt may not have the placeholder_token={self.placeholder_token}")
        # Get the text embedding for e4t conditioning
        encoder_hidden_states_for_e4t = self.text_encoder(input_ids_for_encoder.to(device))[0]
        # Get the text embedding
        inputs_embeds = self.text_encoder.get_input_embeddings()(input_ids.to(device)).to(dtype=self.text_encoder.dtype, device=device)
        return dict(
            placeholder_token_id_idx=placeholder_token_id_idx,
            encoder_hidden_states_for_e4t=encoder_hidden_states_for_e4t,
            inputs_embeds=inputs_embeds
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        ####################################################
        image: Union[PIL.Image.Image, List[PIL.Image.Image]] = None,
        domain_embed_scale: Optional[float] = None,
        ####################################################
    ):
        domain_embed_scale = self.domain_embed_scale if domain_embed_scale is None else domain_embed_scale
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # # 3. Encode input prompt
        # prompt_embeds = self._encode_prompt(
        #     prompt,
        #     device,
        #     num_images_per_prompt,
        #     do_classifier_free_guidance,
        #     negative_prompt,
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=negative_prompt_embeds,
        # )

        # E4t
        assert negative_prompt is None, "negative_prompt is not supported"
        image = preprocess(image)
        e4t_inputs = self.prepare_for_e4t(
            prompt,
            device,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            e4t_inputs["encoder_hidden_states_for_e4t"].dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # E4t
                latents_in = self.scheduler.scale_model_input(latents, t)
                bsz = latents_in.size(0)
                encoder_hidden_states_for_e4t_forward = e4t_inputs["encoder_hidden_states_for_e4t"].expand(bsz, -1, -1)
                # Get the unet encoder outputs
                encoder_outputs = self.unet(latents_in, t, encoder_hidden_states_for_e4t_forward, return_encoder_outputs=True)
                # Forward E4T encoder to get the embedding
                pixel_values = image.expand(bsz, -1, -1, -1).to(device)
                domain_embed = self.e4t_encoder(x=pixel_values, unet_down_block_samples=encoder_outputs["down_block_samples"])
                # update word embedding
                domain_embed = self.class_embed.clone().expand(bsz, -1).to(device) + domain_embed_scale * domain_embed
                inputs_embeds_forward = e4t_inputs["inputs_embeds"].expand(bsz, -1, -1).clone().to(dtype=self.text_encoder.dtype, device=device)
                inputs_embeds_forward[:, e4t_inputs["placeholder_token_id_idx"], :] = domain_embed
                # Get the text embedding for conditioning
                encoder_hidden_states = self.text_encoder(inputs_embeds=inputs_embeds_forward)[0].to(dtype=self.unet.dtype, device=device)
                prompt_embeds = torch.cat([encoder_hidden_states_for_e4t_forward, encoder_hidden_states]) if do_classifier_free_guidance else encoder_hidden_states
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
