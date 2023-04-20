python inference.py \
  --pretrained_model_name_or_path "./output/30/" \
  --prompt "A dog in the style of *s" \
  --num_images_per_prompt 3 \
  --scheduler_type "ddim" \
  --train_image_path="/home/ubuntu/e4t-diffusion/training_images/art/picasso.jpg" \
  --num_inference_steps 50 \
  --guidance_scale 7.5