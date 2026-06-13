#!/usr/bin/env python3
"""Convert SigLIP2 g-opt/16 384 big_vision .npz weights to Transformers format.

Adapted from Hugging Face Transformers'
src/transformers/models/siglip/convert_siglip_to_hf.py.
"""

import argparse
import collections
from pathlib import Path

import numpy as np
import torch
from numpy import load
from transformers import AutoTokenizer, SiglipConfig, SiglipImageProcessor, SiglipModel, SiglipProcessor


MODEL_NAME = "siglip2-giant-opt-patch16-384"
DEFAULT_CHECKPOINT = "/nas/qirui/dinov3/siglip2/siglip2_g-opt16_384.npz"
DEFAULT_OUTPUT_DIR = "/nas/qirui/dinov3/siglip2/siglip2_g-opt16_384_hf"
DEFAULT_TOKENIZER = "google/siglip2-so400m-patch14-384"
MODEL_FILES = ("model.safetensors", "pytorch_model.bin")
PROCESSOR_FILES = ("preprocessor_config.json", "tokenizer_config.json")


def get_siglip2_gopt_384_config():
    text_config = {
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "vocab_size": 256000,
        "projection_size": 1536,
    }
    vision_config = {
        "hidden_size": 1536,
        "intermediate_size": 6144,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "image_size": 384,
        "patch_size": 16,
    }
    return SiglipConfig(text_config=text_config, vision_config=vision_config)


def flatten_nested_dict(params, parent_key="", sep="/"):
    items = []
    for key, value in params.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, collections.abc.MutableMapping):
            items.extend(flatten_nested_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def split_encoderblock_layers(state_dict):
    state_dict = state_dict.copy()
    for key in list(state_dict.keys()):
        if "/encoderblock/" in key:
            weight = state_dict.pop(key)
            for i, weight_i in enumerate(weight):
                state_dict[key.replace("encoderblock", f"encoderblock_{i}")] = weight_i
    return state_dict


def create_rename_keys(config):
    rename_keys = [
        ("params/img/embedding/kernel", "vision_model.embeddings.patch_embedding.weight"),
        ("params/img/embedding/bias", "vision_model.embeddings.patch_embedding.bias"),
        ("params/img/pos_embedding", "vision_model.embeddings.position_embedding.weight"),
    ]

    for i in range(config.vision_config.num_hidden_layers):
        src_prefix = f"params/img/Transformer/encoderblock_{i}"
        dst_prefix = f"vision_model.encoder.layers.{i}"
        rename_keys.extend(
            [
                (f"{src_prefix}/LayerNorm_0/scale", f"{dst_prefix}.layer_norm1.weight"),
                (f"{src_prefix}/LayerNorm_0/bias", f"{dst_prefix}.layer_norm1.bias"),
                (f"{src_prefix}/LayerNorm_1/scale", f"{dst_prefix}.layer_norm2.weight"),
                (f"{src_prefix}/LayerNorm_1/bias", f"{dst_prefix}.layer_norm2.bias"),
                (f"{src_prefix}/MlpBlock_0/Dense_0/kernel", f"{dst_prefix}.mlp.fc1.weight"),
                (f"{src_prefix}/MlpBlock_0/Dense_0/bias", f"{dst_prefix}.mlp.fc1.bias"),
                (f"{src_prefix}/MlpBlock_0/Dense_1/kernel", f"{dst_prefix}.mlp.fc2.weight"),
                (f"{src_prefix}/MlpBlock_0/Dense_1/bias", f"{dst_prefix}.mlp.fc2.bias"),
                (f"{src_prefix}/MultiHeadDotProductAttention_0/key/kernel", f"{dst_prefix}.self_attn.k_proj.weight"),
                (f"{src_prefix}/MultiHeadDotProductAttention_0/key/bias", f"{dst_prefix}.self_attn.k_proj.bias"),
                (f"{src_prefix}/MultiHeadDotProductAttention_0/value/kernel", f"{dst_prefix}.self_attn.v_proj.weight"),
                (f"{src_prefix}/MultiHeadDotProductAttention_0/value/bias", f"{dst_prefix}.self_attn.v_proj.bias"),
                (f"{src_prefix}/MultiHeadDotProductAttention_0/query/kernel", f"{dst_prefix}.self_attn.q_proj.weight"),
                (f"{src_prefix}/MultiHeadDotProductAttention_0/query/bias", f"{dst_prefix}.self_attn.q_proj.bias"),
                (f"{src_prefix}/MultiHeadDotProductAttention_0/out/kernel", f"{dst_prefix}.self_attn.out_proj.weight"),
                (f"{src_prefix}/MultiHeadDotProductAttention_0/out/bias", f"{dst_prefix}.self_attn.out_proj.bias"),
            ]
        )

    rename_keys.extend(
        [
            ("params/img/Transformer/encoder_norm/scale", "vision_model.post_layernorm.weight"),
            ("params/img/Transformer/encoder_norm/bias", "vision_model.post_layernorm.bias"),
            ("params/img/MAPHead_0/probe", "vision_model.head.probe"),
            ("params/img/MAPHead_0/LayerNorm_0/scale", "vision_model.head.layernorm.weight"),
            ("params/img/MAPHead_0/LayerNorm_0/bias", "vision_model.head.layernorm.bias"),
            ("params/img/MAPHead_0/MlpBlock_0/Dense_0/kernel", "vision_model.head.mlp.fc1.weight"),
            ("params/img/MAPHead_0/MlpBlock_0/Dense_0/bias", "vision_model.head.mlp.fc1.bias"),
            ("params/img/MAPHead_0/MlpBlock_0/Dense_1/kernel", "vision_model.head.mlp.fc2.weight"),
            ("params/img/MAPHead_0/MlpBlock_0/Dense_1/bias", "vision_model.head.mlp.fc2.bias"),
            ("params/img/MAPHead_0/MultiHeadDotProductAttention_0/out/kernel", "vision_model.head.attention.out_proj.weight"),
            ("params/img/MAPHead_0/MultiHeadDotProductAttention_0/out/bias", "vision_model.head.attention.out_proj.bias"),
            ("params/txt/Embed_0/embedding", "text_model.embeddings.token_embedding.weight"),
            ("params/txt/pos_embedding", "text_model.embeddings.position_embedding.weight"),
        ]
    )

    for i in range(config.text_config.num_hidden_layers):
        src_prefix = f"params/txt/Encoder_0/encoderblock_{i}"
        dst_prefix = f"text_model.encoder.layers.{i}"
        rename_keys.extend(
            [
                (f"{src_prefix}/LayerNorm_0/scale", f"{dst_prefix}.layer_norm1.weight"),
                (f"{src_prefix}/LayerNorm_0/bias", f"{dst_prefix}.layer_norm1.bias"),
                (f"{src_prefix}/LayerNorm_1/scale", f"{dst_prefix}.layer_norm2.weight"),
                (f"{src_prefix}/LayerNorm_1/bias", f"{dst_prefix}.layer_norm2.bias"),
                (f"{src_prefix}/MlpBlock_0/Dense_0/kernel", f"{dst_prefix}.mlp.fc1.weight"),
                (f"{src_prefix}/MlpBlock_0/Dense_0/bias", f"{dst_prefix}.mlp.fc1.bias"),
                (f"{src_prefix}/MlpBlock_0/Dense_1/kernel", f"{dst_prefix}.mlp.fc2.weight"),
                (f"{src_prefix}/MlpBlock_0/Dense_1/bias", f"{dst_prefix}.mlp.fc2.bias"),
                (f"{src_prefix}/MultiHeadDotProductAttention_0/key/kernel", f"{dst_prefix}.self_attn.k_proj.weight"),
                (f"{src_prefix}/MultiHeadDotProductAttention_0/key/bias", f"{dst_prefix}.self_attn.k_proj.bias"),
                (f"{src_prefix}/MultiHeadDotProductAttention_0/value/kernel", f"{dst_prefix}.self_attn.v_proj.weight"),
                (f"{src_prefix}/MultiHeadDotProductAttention_0/value/bias", f"{dst_prefix}.self_attn.v_proj.bias"),
                (f"{src_prefix}/MultiHeadDotProductAttention_0/query/kernel", f"{dst_prefix}.self_attn.q_proj.weight"),
                (f"{src_prefix}/MultiHeadDotProductAttention_0/query/bias", f"{dst_prefix}.self_attn.q_proj.bias"),
                (f"{src_prefix}/MultiHeadDotProductAttention_0/out/kernel", f"{dst_prefix}.self_attn.out_proj.weight"),
                (f"{src_prefix}/MultiHeadDotProductAttention_0/out/bias", f"{dst_prefix}.self_attn.out_proj.bias"),
            ]
        )

    rename_keys.extend(
        [
            ("params/txt/Encoder_0/encoder_norm/scale", "text_model.final_layer_norm.weight"),
            ("params/txt/Encoder_0/encoder_norm/bias", "text_model.final_layer_norm.bias"),
            ("params/txt/head/kernel", "text_model.head.weight"),
            ("params/txt/head/bias", "text_model.head.bias"),
            ("params/t", "logit_scale"),
            ("params/b", "logit_bias"),
        ]
    )
    return rename_keys


def rename_key(state_dict, old, new, config):
    if old not in state_dict:
        raise KeyError(f"Missing expected checkpoint key: {old}")

    val = state_dict.pop(old)
    if ("out_proj" in new or "v_proj" in new or "k_proj" in new or "q_proj" in new) and "vision" in new:
        val = val.reshape(-1, config.vision_config.hidden_size)
    if ("out_proj" in new or "v_proj" in new or "k_proj" in new or "q_proj" in new) and "text" in new:
        val = val.reshape(-1, config.text_config.hidden_size)

    if "patch_embedding.weight" in new:
        val = val.transpose(3, 2, 0, 1)
    elif new.endswith("weight") and "position_embedding" not in new and "token_embedding" not in new:
        val = val.T

    if "position_embedding" in new and "vision" in new:
        val = val.reshape(-1, config.vision_config.hidden_size)
    if "position_embedding" in new and "text" in new:
        val = val.reshape(-1, config.text_config.hidden_size)
    if new.endswith("bias"):
        val = val.reshape(-1)

    state_dict[new] = torch.from_numpy(val)


def read_in_q_k_v_head(state_dict, config):
    prefix = "params/img/MAPHead_0/MultiHeadDotProductAttention_0"
    key_proj_weight = state_dict.pop(f"{prefix}/key/kernel").reshape(-1, config.vision_config.hidden_size).T
    key_proj_bias = state_dict.pop(f"{prefix}/key/bias").reshape(-1)
    value_proj_weight = state_dict.pop(f"{prefix}/value/kernel").reshape(-1, config.vision_config.hidden_size).T
    value_proj_bias = state_dict.pop(f"{prefix}/value/bias").reshape(-1)
    query_proj_weight = state_dict.pop(f"{prefix}/query/kernel").reshape(-1, config.vision_config.hidden_size).T
    query_proj_bias = state_dict.pop(f"{prefix}/query/bias").reshape(-1)

    state_dict["vision_model.head.attention.in_proj_weight"] = torch.from_numpy(
        np.concatenate([query_proj_weight, key_proj_weight, value_proj_weight], axis=0)
    )
    state_dict["vision_model.head.attention.in_proj_bias"] = torch.from_numpy(
        np.concatenate([query_proj_bias, key_proj_bias, value_proj_bias], axis=0)
    )


@torch.no_grad()
def convert_siglip2_gopt_384(checkpoint_path, output_dir, tokenizer_name=DEFAULT_TOKENIZER):
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_exists = (output_dir / "config.json").exists() and any((output_dir / name).exists() for name in MODEL_FILES)

    if model_exists:
        print(f"Model files already exist in {output_dir}; only ensuring processor/tokenizer files.")
    else:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        print(f"Loading checkpoint: {checkpoint_path}")
        config = get_siglip2_gopt_384_config()
        data = load(checkpoint_path)
        state_dict = split_encoderblock_layers(flatten_nested_dict(data))

        for src, dest in create_rename_keys(config):
            rename_key(state_dict, src, dest, config)
        read_in_q_k_v_head(state_dict, config)

        print("Building Transformers SiglipModel")
        model = SiglipModel(config).eval()
        model.load_state_dict(state_dict)

        print(f"Saving model to: {output_dir}")
        model.save_pretrained(output_dir)

    print(f"Loading tokenizer from: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        add_bos_token=False,
        add_eos_token=True,
        padding_side="right",
        do_lower_case=True,
        model_input_names=["input_ids"],
    )
    image_processor = SiglipImageProcessor(size={"height": 384, "width": 384}, resample=2)
    processor = SiglipProcessor(image_processor=image_processor, tokenizer=tokenizer)
    processor.save_pretrained(output_dir)

    missing = [name for name in PROCESSOR_FILES if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Processor save did not create required files: {missing}")

    print("Conversion complete")
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description=f"Convert {MODEL_NAME} .npz to Transformers format")
    parser.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer_name", default=DEFAULT_TOKENIZER)
    args = parser.parse_args()
    convert_siglip2_gopt_384(args.checkpoint_path, args.output_dir, args.tokenizer_name)


if __name__ == "__main__":
    main()
