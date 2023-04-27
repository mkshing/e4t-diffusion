INPUT_PATH=$1
PROMPT=$2
INPUT_PATH="/home/ubuntu/e4t-diffusion/training_images/$INPUT_PATH"
PROJECT="diffusiondb"

echo "Prompt: $PROMPT"
echo "Input path: $INPUT_PATH"

python inference.py \
  --pretrained_model_name_or_path "./output/$PROJECT/100/" \
  --prompt "$PROMPT" \
  --num_images_per_prompt 8 \
  --scheduler_type "ddim" \
  --output_dir "./output/$PROJECT" \
  --image_path_or_url="$INPUT_PATH" \
  --num_inference_steps 50 \
  --guidance_scale 7.5