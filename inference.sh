PROMPT=$1
INPUT_PATH=$2
INPUT_PATH="/home/ubuntu/e4t-diffusion/training_images/$INPUT_PATH"

echo "Prompt: $PROMPT"
echo "Input path: $INPUT_PATH"

python inference.py \
  --pretrained_model_name_or_path "./output/30/" \
  --prompt "$PROMPT" \
  --num_images_per_prompt 3 \
  --scheduler_type "ddim" \
  --image_path_or_url="$INPUT_PATH" \
  --num_inference_steps 50 \
  --guidance_scale 7.5