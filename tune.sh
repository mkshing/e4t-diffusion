accelerate launch tuning_e4t.py \
  --pretrained_model_name_or_path="./pretrained_wikiart" \
  --prompt_template="a dog in the style of {placeholder_token}" \
  --reg_lambda=0.1 \
  --output_dir="./output" \
  --train_image_path="./training_images/isometric/isometric1.png" \
  --resolution=256 \
  --train_batch_size=16 \
  --learning_rate=1e-6 --scale_lr \
  --max_train_steps=30 \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention
