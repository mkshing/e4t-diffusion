# E4T-diffusion
<!-- <a href=".." target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> -->

An implementation of [Encoder-based Domain Tuning for Fast Personalization of Text-to-Image Models](https://arxiv.org/abs/2302.12228) by using d🧨ffusers. 

My summary tweet is found [here](https://twitter.com/mk1stats/status/1630891691623448576).

![paper](e4t-paper.png)

## Installation
```
$ git clone https://github.com/mkshing/e4t-diffusion.git
$ cd e4t-diffusion
$ pip install -r requirements.txt
```

## Pre-training
```
accelerate launch pretrain_e4t.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --clip_model_name_or_path="ViT-H-14::laion2b_s32b_b79k" \
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
  --train_batch_size=16 \
  --learning_rate=1e-6 --scale_lr \
  --checkpointing_steps=10000 \
  --log_steps=1000 \
  --max_train_steps=100000 \
  --unfreeze_clip_vision \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention 
```

## Domain-tuning

```
accelerate launch tuning_e4t.py \
  --pretrained_model_name_or_path="e4t pre-trained model path" \
  --reg_lambda=0.1 \
  --output_dir="path-to-save-model" \
  --train_image_path="image path or url" \
  --resolution=512 \
  --train_batch_size=16 \
  --learning_rate=5.e-6 \
  --max_train_steps=15 \
  --unfreeze_clip_vision \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention 
```

## Inference
```
python inference.py \
  --pretrained_model_name_or_path "e4t pre-trained model path" \
  --prompt "Times square in the style of *s" \
  --num_images_per_prompt 3 \
  --scheduler_type "ddim" \
  --image_path_or_url "same image path or url as domain tuning" \
  --num_inference_steps 50 \
  --guidance_scale 7.5
```


## Citation

```bibtex
@misc{https://doi.org/10.48550/arXiv.2302.12228,
    url     = {https://arxiv.org/abs/2302.12228},
    author  = {Rinon Gal, Moab Arar, Yuval Atzmon, Amit H. Bermano, Gal Chechik, Daniel Cohen-Or},  
    title   = {Encoder-based Domain Tuning for Fast Personalization of Text-to-Image Models},
    publisher = {arXiv},
    year    = {2023},
    copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## TODO
- [x] Pre-training
- [x] Domain-tuning
- [x] Inference
- [x] Data augmentation by [stable unclip](https://github.com/Stability-AI/stablediffusion)
- [ ] Use an off-the-shelf face segmentation network for human face domain.
   > Finally, we find that for the human face domain, it is helpful to
use an off-the-shelf face segmentation network [Deng et al. 2019]
to mask the diffusion loss at this stage.
