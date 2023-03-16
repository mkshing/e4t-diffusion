"""
Allow to input embeddings
"""
import torch
from transformers import CLIPTextModel as OriginalCLIPTextModel
from transformers.models.clip.modeling_clip import _expand_mask, BaseModelOutputWithPooling


class CLIPTextModel(OriginalCLIPTextModel):
    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_attentions = output_attentions if output_attentions is not None else self.text_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.text_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.text_model.config.use_return_dict

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify input_ids or inputs_embeds")

        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            bsz, seq_len = input_shape
        else:
            bsz, seq_len, _ = inputs_embeds.size()
        hidden_states = self.text_model.embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids
        )

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        # pooled_output = last_hidden_state[
        #     torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        #     input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        # ]
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), 0]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


if __name__ == '__main__':
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    inputs_embeds = torch.randn(2, 77, 512)
    outputs = model(inputs_embeds=inputs_embeds)