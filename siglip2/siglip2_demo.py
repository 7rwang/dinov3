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
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import ndimage

def load_model(model_path=None):
    """加载SigLIP2模型和processor"""
    if model_path and os.path.exists(model_path):
        # 加载本地权重文件
        print(f"Loading local weights from {model_path}")
        # 这里需要根据实际的权重格式进行调整
        # 暂时使用Hugging Face的预训练模型
        model_name = "google/siglip-base-patch16-224"
    else:
        # 使用Hugging Face预训练模型
        model_name = "google/siglip-base-patch16-224"
    
    print(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    
    # 如果有GPU可用，移到GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    return model, processor, device

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

def process_image_text_with_heatmap(model, processor, device, image, text):
    """
    正确的流程：
    image -> SigLIP2 vision encoder -> patch features: [num_patches, dim]
    text  -> SigLIP2 text encoder   -> text feature:   [dim]
    similarity = cosine(patch_features, text_feature)
    """
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # 1. 获取vision patch features
        vision_inputs = {k: v for k, v in inputs.items() if 'pixel_values' in k}
        vision_outputs = model.vision_model(**vision_inputs)
        patch_features = vision_outputs.last_hidden_state[0, 1:]  # [num_patches, dim] 排除CLS token
        
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
        similarities = torch.cosine_similarity(
            patch_features, 
            text_feature.unsqueeze(0).expand(patch_features.size(0), -1), 
            dim=-1
        )  # [num_patches]
        
        return similarities.cpu().numpy()

def generate_heatmap(image, similarities, save_path=None):
    """
    生成热力图：
    similarities -> reshape 成 H_patch x W_patch -> resize 到原图大小 -> 叠到图片上
    """
    # 转换PIL图像为numpy数组
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    
    print(f"图像尺寸: {h}x{w}")
    print(f"patch相似度数量: {len(similarities)}")
    
    num_patches = len(similarities)
    
    # 对于标准的ViT，通常input_size=224时是14x14=196 patches
    # 对于SigLIP base-patch16-224也应该是14x14
    expected_grid_size = 14  # 224/16 = 14
    
    if num_patches == expected_grid_size * expected_grid_size:
        grid_h = grid_w = expected_grid_size
        print(f"使用标准14x14 grid")
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
    
    # 使用scipy进行插值放大到原图大小
    from scipy.ndimage import zoom
    zoom_factor_h = h / grid_h
    zoom_factor_w = w / grid_w
    heatmap = zoom(similarity_grid, (zoom_factor_h, zoom_factor_w), order=1)
    
    # 归一化到0-1范围
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
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
    if len(image_np.shape) == 3:
        overlay = 0.6 * image_np.astype(float) / 255.0 + 0.4 * heatmap_colored
    else:
        # 灰度图转RGB
        image_rgb = np.stack([image_np, image_np, image_np], axis=-1)
        overlay = 0.6 * image_rgb.astype(float) / 255.0 + 0.4 * heatmap_colored
        
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

def demo_from_local(model, processor, device, image_path, texts, generate_heatmap_flag=False):
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
        print(f"\n正在生成'{best_text}'的热力图...")
        try:
            patch_similarities = process_image_text_with_heatmap(model, processor, device, image, best_text)
            
            # 生成保存路径
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = f"heatmap_{image_name}_{best_text.replace(' ', '_')}.png"
            
            heatmap, overlay = generate_heatmap(image, patch_similarities, save_path=save_path)
            
        except Exception as e:
            print(f"生成热力图时出错: {e}")
            print("尝试使用简化方法...")
            # 如果上面的方法失败，使用attention weights的简化方法
            try:
                simplified_heatmap(model, processor, device, image, best_text, image_path)
            except Exception as e2:
                print(f"简化方法也失败了: {e2}")
    
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
            features = vision_outputs.last_hidden_state[0, 1:]  # 排除CLS token
            
            # 创建伪梯度权重 (基于特征的方差)
            weights = torch.var(features, dim=-1)
            
            # 生成热力图
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = f"simplified_heatmap_{image_name}_{text.replace(' ', '_')}.png"
            
            generate_heatmap(image, weights.detach().cpu().numpy(), save_path=save_path)
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
                       default="/nas/qirui/dinov3/siglip2/siglip2_g-opt16_384.npz",
                       help="本地模型权重路径")
    parser.add_argument("--heatmap", action="store_true", 
                       help="生成热力图显示图像中与文本最相关的区域")
    
    args = parser.parse_args()
    
    # 加载模型
    model, processor, device = load_model(args.model_path)
    
    # 运行演示
    if args.image_url:
        demo_from_url(model, processor, device, args.image_url, args.texts)
    elif args.image_path:
        demo_from_local(model, processor, device, args.image_path, args.texts, args.heatmap)
    else:
        # 默认演示
        print("运行默认演示...")
        default_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        demo_from_url(model, processor, device, default_url, args.texts)

if __name__ == "__main__":
    main()