#!/usr/bin/env python3
"""
SigLIP2 Demo Script
使用SigLIP2 g-opt/16 384模型进行图像-文本匹配
"""

import torch
import numpy as np
from PIL import Image
import requests
from transformers import AutoModel, AutoProcessor
import argparse
import importlib.util
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn.functional as F

DEFAULT_HF_MODEL = "google/siglip2-so400m-patch14-384"
DEFAULT_NPZ_MODEL = "/nas/qirui/dinov3/siglip2/siglip2_g-opt16_384.npz"

def convert_npz_to_transformers_dir(model_path, output_dir=None):
    checkpoint_path = Path(model_path)
    if output_dir is None:
        output_dir = checkpoint_path.with_name(f"{checkpoint_path.stem}_hf")
    output_dir = Path(output_dir)

    required_files = ("config.json", "preprocessor_config.json", "tokenizer_config.json")
    if all((output_dir / filename).exists() for filename in required_files):
        print(f"Using converted Transformers model: {output_dir}")
        return str(output_dir)

    converter_path = Path(__file__).with_name("convert_siglip2_npz_to_hf.py")
    spec = importlib.util.spec_from_file_location("convert_siglip2_npz_to_hf", converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)

    print(f"Converting {checkpoint_path} to Transformers format at {output_dir}")
    return converter.convert_siglip2_gopt_384(str(checkpoint_path), str(output_dir))

def load_model(model_path=None, hf_model_name=DEFAULT_HF_MODEL):
    """加载SigLIP2模型和processor"""
    if model_path:
        if os.path.isdir(model_path):
            model_name = model_path
        elif model_path.endswith(".npz") and os.path.exists(model_path):
            model_name = convert_npz_to_transformers_dir(model_path)
        elif os.path.exists(model_path):
            raise ValueError(f"Unsupported model_path format: {model_path}")
        else:
            print(f"Model path does not exist: {model_path}")
            print(f"Falling back to Hugging Face model: {hf_model_name}")
            model_name = hf_model_name
    else:
        model_name = hf_model_name
    
    print(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    
    # 如果有GPU可用，移到GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"Model loaded on {device}")
    
    return model, processor, device

def _get_vision_image_size(model, pixel_values):
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    if vision_config is not None and hasattr(vision_config, "image_size"):
        image_size = vision_config.image_size
        if isinstance(image_size, (tuple, list)):
            return int(image_size[0]), int(image_size[1])
        return int(image_size), int(image_size)
    return int(pixel_values.shape[-2]), int(pixel_values.shape[-1])

def _get_patch_size(model):
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    if vision_config is not None and hasattr(vision_config, "patch_size"):
        patch_size = vision_config.patch_size
        if isinstance(patch_size, (tuple, list)):
            return int(patch_size[0]), int(patch_size[1])
        return int(patch_size), int(patch_size)
    return None

def _infer_patch_grid(model, pixel_values, num_patches):
    image_h, image_w = _get_vision_image_size(model, pixel_values)
    patch_size = _get_patch_size(model)
    if patch_size is not None:
        patch_h, patch_w = patch_size
        grid_h, grid_w = image_h // patch_h, image_w // patch_w
        if grid_h * grid_w == num_patches:
            return grid_h, grid_w

    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size == num_patches:
        return grid_size, grid_size

    raise ValueError(
        f"Cannot infer patch grid for {num_patches} patches. "
        "Check whether the model output contains extra tokens."
    )

def process_image_text(model, processor, device, image, texts):
    """处理图像和文本，返回相似度分数"""
    # 处理输入
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 获取嵌入
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
    
    # 计算相似度
    similarities = torch.cosine_similarity(text_embeds, image_embeds.unsqueeze(0), dim=-1)
    return similarities.cpu().numpy()

def score_image_text_logits(model, processor, device, images, text):
    """Return SigLIP logits for one text against one or more images."""
    single_image = not isinstance(images, (list, tuple))
    image_batch = [images] if single_image else list(images)
    inputs = processor(text=[text], images=image_batch, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits_per_image[:, 0]

    scores = scores.detach().cpu().numpy()
    return scores[0] if single_image else scores

def process_image_text_with_heatmap(model, processor, device, image, text):
    """
    Fast but approximate patch-token heatmap.

    SigLIP2 is trained with an image-level head, so patch-token cosine is only a
    rough diagnostic. Use occlusion_heatmap for a more faithful localization.
    """
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # 1. 获取vision patch features
        vision_inputs = {k: v for k, v in inputs.items() if 'pixel_values' in k}
        vision_outputs = model.vision_model(**vision_inputs)
        patch_features = vision_outputs.last_hidden_state[0]
        
        # 2. 获取text feature
        text_inputs = {k: v for k, v in inputs.items() if 'input_ids' in k or 'attention_mask' in k}
        text_outputs = model.text_model(**text_inputs)
        text_feature = text_outputs.pooler_output[0]  # [dim]
        
        # 3. 投影到相同空间
        if hasattr(model, 'visual_projection'):
            patch_features = model.visual_projection(patch_features)  # [num_patches, proj_dim]
        if hasattr(model, 'text_projection'):
            text_feature = model.text_projection(text_feature)  # [proj_dim]
        
        # 4. 计算cosine similarity
        patch_features = F.normalize(patch_features, dim=-1)
        text_feature = F.normalize(text_feature, dim=-1)
        similarities = patch_features @ text_feature
        
        grid_shape = _infer_patch_grid(model, inputs["pixel_values"], patch_features.shape[0])
        input_size = tuple(inputs["pixel_values"].shape[-2:])
        return similarities.cpu().numpy(), grid_shape, input_size

def process_image_text_with_occlusion_heatmap(
    model,
    processor,
    device,
    image,
    text,
    grid_size=16,
    batch_size=4,
):
    """Generate a heatmap by measuring how much each occlusion lowers the text logit."""
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    input_h, input_w = tuple(inputs["pixel_values"].shape[-2:])
    resized_image = image.convert("RGB").resize((input_w, input_h), Image.BICUBIC)

    baseline_score = score_image_text_logits(model, processor, device, resized_image, text)
    print(f"Baseline logit for '{text}': {baseline_score:.4f}")

    image_np = np.array(resized_image)
    fill_color = np.mean(image_np.reshape(-1, 3), axis=0).astype(np.uint8)
    row_edges = np.linspace(0, input_h, grid_size + 1, dtype=int)
    col_edges = np.linspace(0, input_w, grid_size + 1, dtype=int)

    occluded_images = []
    locations = []
    for row in range(grid_size):
        for col in range(grid_size):
            occluded = image_np.copy()
            y0, y1 = row_edges[row], row_edges[row + 1]
            x0, x1 = col_edges[col], col_edges[col + 1]
            occluded[y0:y1, x0:x1] = fill_color
            occluded_images.append(Image.fromarray(occluded))
            locations.append((row, col))

    heatmap_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for start in range(0, len(occluded_images), batch_size):
        end = min(start + batch_size, len(occluded_images))
        batch_scores = score_image_text_logits(
            model,
            processor,
            device,
            occluded_images[start:end],
            text,
        )
        for score, (row, col) in zip(batch_scores, locations[start:end]):
            heatmap_grid[row, col] = max(0.0, baseline_score - score)

    return heatmap_grid.reshape(-1), (grid_size, grid_size), (input_h, input_w)

def generate_heatmap(image, similarities, grid_shape=None, input_size=None, save_path=None):
    """
    生成热力图：
    similarities -> reshape 成 H_patch x W_patch -> resize 到原图大小 -> 叠到图片上
    """
    # 转换PIL图像为numpy数组
    if input_size is not None:
        input_h, input_w = input_size
        image = image.convert("RGB").resize((input_w, input_h), Image.BICUBIC)
    else:
        image = image.convert("RGB")

    image_np = np.array(image)
    h, w = image_np.shape[:2]
    
    print(f"图像尺寸: {h}x{w}")
    print(f"patch相似度数量: {len(similarities)}")
    
    num_patches = len(similarities)
    
    if grid_shape is not None:
        grid_h, grid_w = grid_shape
        if grid_h * grid_w != num_patches:
            raise ValueError(f"grid_shape {grid_shape} does not match {num_patches} patches")
        print(f"使用模型grid: {grid_h}x{grid_w}")
    else:
        # 尝试找到最接近正方形的因数分解
        grid_size = int(np.sqrt(num_patches))
        if grid_size * grid_size == num_patches:
            grid_h = grid_w = grid_size
            print(f"使用{grid_size}x{grid_size} 正方形grid")
        else:
            # 找到最佳的矩形分解
            best_ratio = float('inf')
            best_h, best_w = 1, num_patches
            
            for h_try in range(1, int(np.sqrt(num_patches)) + 1):
                if num_patches % h_try == 0:
                    w_try = num_patches // h_try
                    ratio = max(h_try/w_try, w_try/h_try)  # 长宽比
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_h, best_w = h_try, w_try
            
            grid_h, grid_w = best_h, best_w
            print(f"使用{grid_h}x{grid_w} 矩形grid (长宽比: {best_ratio:.2f})")
    
    # 重塑为2D网格
    similarity_grid = similarities.reshape(grid_h, grid_w)
    
    heatmap_tensor = torch.from_numpy(similarity_grid).float()[None, None]
    heatmap = F.interpolate(heatmap_tensor, size=(h, w), mode="bilinear", align_corners=False)
    heatmap = heatmap[0, 0].numpy()
    
    # 归一化到0-1范围
    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()
    if heatmap_max > heatmap_min:
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap = np.zeros_like(heatmap)
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 热力图
    im = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Similarity Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # 叠加图
    # 将热力图转换为RGB
    heatmap_colored = cm.jet(heatmap)[:,:,:3]
    
    # 叠加
    overlay = 0.6 * image_np.astype(float) / 255.0 + 0.4 * heatmap_colored
        
    overlay = np.clip(overlay, 0, 1)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"热力图已保存到: {save_path}")
    
    plt.show()
    
    return heatmap, overlay

def demo_from_url(model, processor, device, image_url, texts):
    """从URL加载图像并进行匹配"""
    print(f"Loading image from: {image_url}")
    response = requests.get(image_url)
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    similarities = process_image_text(model, processor, device, image, texts)
    
    print("\n--- 图像-文本匹配结果 ---")
    for text, score in zip(texts, similarities):
        print(f"{text}: {score.item():.4f}")
    
    # 找到最匹配的文本
    best_idx = np.argmax(similarities)
    print(f"\n最匹配的描述: '{texts[best_idx]}' (得分: {similarities[best_idx].item():.4f})")
    
    return similarities

def demo_from_local(
    model,
    processor,
    device,
    image_path,
    texts,
    generate_heatmap_flag=False,
    heatmap_method="occlusion",
    heatmap_text=None,
    occlusion_grid=16,
    occlusion_batch_size=4,
):
    """从本地文件加载图像并进行匹配"""
    print(f"Loading local image: {image_path}")
    image = Image.open(image_path)
    
    similarities = process_image_text(model, processor, device, image, texts)
    
    print("\n--- 图像-文本匹配结果 ---")
    for text, score in zip(texts, similarities):
        print(f"{text}: {score.item():.4f}")
    
    # 找到最匹配的文本
    best_idx = np.argmax(similarities)
    best_text = texts[best_idx]
    print(f"\n最匹配的描述: '{best_text}' (得分: {similarities[best_idx].item():.4f})")
    
    # 生成热力图
    if generate_heatmap_flag:
        target_text = heatmap_text or best_text
        print(f"\n正在用 {heatmap_method} 方法生成'{target_text}'的热力图...")
        try:
            if heatmap_method == "occlusion":
                patch_similarities, grid_shape, input_size = process_image_text_with_occlusion_heatmap(
                    model,
                    processor,
                    device,
                    image,
                    target_text,
                    grid_size=occlusion_grid,
                    batch_size=occlusion_batch_size,
                )
            else:
                patch_similarities, grid_shape, input_size = process_image_text_with_heatmap(
                    model, processor, device, image, target_text
                )
            
            # 生成保存路径
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            safe_text = target_text.replace(" ", "_").replace("/", "_")
            save_path = f"heatmap_{heatmap_method}_{image_name}_{safe_text}.png"
            
            heatmap, overlay = generate_heatmap(
                image,
                patch_similarities,
                grid_shape=grid_shape,
                input_size=input_size,
                save_path=save_path,
            )
            
        except Exception as e:
            print(f"生成热力图时出错: {e}")
    
    return similarities

def simplified_heatmap(model, processor, device, image, text, image_path):
    """简化的热力图生成方法 - 使用梯度激活映射"""
    print("使用Grad-CAM方法生成热力图...")
    
    # 获取模型输出并计算梯度
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 启用梯度计算
    for param in model.parameters():
        param.requires_grad = True
        
    model.eval()
    
    try:
        # 前向传播
        outputs = model(**inputs)
        
        # 获取logits并计算目标分数 
        logits_per_image = outputs.logits_per_image
        target_score = logits_per_image[0, 0]  # 第一个文本的分数
        
        # 反向传播计算梯度
        model.zero_grad()
        target_score.backward(retain_graph=True)
        
        # 尝试获取vision模型的特征图和梯度
        if hasattr(model, 'vision_model'):
            # 获取最后的卷积层或attention层
            vision_model = model.vision_model
            
            # 简化方法：使用图像的patch embeddings
            vision_outputs = vision_model(inputs['pixel_values'], output_hidden_states=True)
            features = vision_outputs.last_hidden_state[0]
            
            # 创建伪梯度权重 (基于特征的方差)
            weights = torch.var(features, dim=-1)
            
            # 生成热力图
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = f"simplified_heatmap_{image_name}_{text.replace(' ', '_')}.png"
            
            grid_shape = _infer_patch_grid(model, inputs["pixel_values"], features.shape[0])
            input_size = tuple(inputs["pixel_values"].shape[-2:])
            generate_heatmap(
                image,
                weights.detach().cpu().numpy(),
                grid_shape=grid_shape,
                input_size=input_size,
                save_path=save_path,
            )
        else:
            print("无法访问vision模型特征，使用随机热力图演示...")
            # 生成随机热力图作为演示
            random_similarities = np.random.random(196)  # 14x14 patches
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = f"demo_heatmap_{image_name}_{text.replace(' ', '_')}.png"
            generate_heatmap(image, random_similarities, save_path=save_path)
            
    except Exception as e:
        print(f"Grad-CAM方法失败: {e}")
        print("生成演示热力图...")
        # 生成基于图像内容的演示热力图
        demo_heatmap_from_image(image, text, image_path)

def demo_heatmap_from_image(image, text, image_path):
    """基于图像内容生成演示热力图"""
    import numpy as np
    from PIL import Image
    
    # 将图像转换为灰度并调整大小
    image_gray = image.convert('L')
    image_resized = image_gray.resize((14, 14))  # 模拟14x14 patches
    
    # 使用图像强度作为"相似度"
    similarities = np.array(image_resized).flatten()
    similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())
    
    # 添加一些随机变化
    similarities += np.random.normal(0, 0.1, similarities.shape)
    similarities = np.clip(similarities, 0, 1)
    
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = f"demo_heatmap_{image_name}_{text.replace(' ', '_')}.png"
    
    print(f"生成基于图像内容的演示热力图...")
    generate_heatmap(image, similarities, save_path=save_path)

def main():
    parser = argparse.ArgumentParser(description="SigLIP2 Demo")
    parser.add_argument("--image_url", type=str, help="图像URL")
    parser.add_argument("--image_path", type=str, help="本地图像路径")
    parser.add_argument("--texts", type=str, nargs="+", 
                       default=["a cat", "a dog", "a bird", "a car", "a person"],
                       help="文本描述列表")
    parser.add_argument("--model_path", type=str, 
                       default=DEFAULT_NPZ_MODEL,
                       help="本地模型权重路径")
    parser.add_argument("--hf_model_name", type=str,
                       default=DEFAULT_HF_MODEL,
                       help="Transformers模型名或本地Transformers模型目录")
    parser.add_argument("--heatmap", action="store_true", 
                       help="生成热力图显示图像中与文本最相关的区域")
    parser.add_argument("--heatmap_method", choices=["occlusion", "patch_similarity"],
                       default="occlusion",
                       help="热力图方法：occlusion更慢但更可信；patch_similarity更快但只是粗略诊断")
    parser.add_argument("--heatmap_text", type=str,
                       help="指定生成热力图的文本；默认使用匹配分数最高的文本")
    parser.add_argument("--occlusion_grid", type=int, default=16,
                       help="occlusion heatmap网格大小，默认16x16")
    parser.add_argument("--occlusion_batch_size", type=int, default=4,
                       help="occlusion前向传播batch size，显存不够时调小")
    
    args = parser.parse_args()
    
    # 加载模型
    model, processor, device = load_model(args.model_path, args.hf_model_name)
    
    # 运行演示
    if args.image_url:
        demo_from_url(model, processor, device, args.image_url, args.texts)
    elif args.image_path:
        demo_from_local(
            model,
            processor,
            device,
            args.image_path,
            args.texts,
            args.heatmap,
            heatmap_method=args.heatmap_method,
            heatmap_text=args.heatmap_text,
            occlusion_grid=args.occlusion_grid,
            occlusion_batch_size=args.occlusion_batch_size,
        )
    else:
        # 默认演示
        print("运行默认演示...")
        default_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        demo_from_url(model, processor, device, default_url, args.texts)

if __name__ == "__main__":
    main()
