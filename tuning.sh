INPUT_PATH=$1
INPUT_PATH="/home/ubuntu/e4t-diffusion/training_images/$INPUT_PATH"
PROJECT="diffusiondb"

accelerate launch tuning_e4t.py \
  --pretrained_model_name_or_path="/home/ubuntu/e4t-diffusion/pretrained-diffusiondb/100000/" \
  --reg_lambda=1e-4 \
  --output_dir "./output/$PROJECT" \
  --train_image_path="$INPUT_PATH" \
  --resolution=256 \
  --train_batch_size=4 \
  --learning_rate=1e-6 \
  --scale_lr \
  --max_train_steps=100 \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention