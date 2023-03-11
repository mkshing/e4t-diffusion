from transformers import CLIPTokenizer


pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
domain_class_token = "pokemon"
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
domain_class_token_id = tokenizer(domain_class_token, add_special_tokens=False, return_tensors="pt").input_ids[0]
assert domain_class_token_id.size(0) == 1, domain_class_token_id
