accelerate launch pretrain_e4t.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --domain_class_token="art" \
  --placeholder_token="*s" \
  --prompt_template="art" \
  --save_sample_prompt="a photo of the *s,a photo of the *s in monet style" \
  --reg_lambda=0.01 \
  --domain_embed_scale=0.1 \
  --output_dir="pretrained-wikiart" \
  --train_image_dataset="Artificio/WikiArt" \
  --iterable_dataset \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-6 --scale_lr \
  --checkpointing_steps=10000 \
  --log_steps=1000 \
  --max_train_steps=100000 \
  --unfreeze_clip_vision \
  --enable_xformers_memory_efficient_attention 