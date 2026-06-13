# SigLIP-2 Installation Manual for DINOv3 Environment

## Environment Overview
- **Current Environment**: `dinov3` (active)
- **Python Version**: 3.11.14
- **Location**: `/nas/qirui/.conda/envs/dinov3`
- **Package Manager**: conda + pip

## Current Dependencies Status
✅ **Already Installed:**
- `torch==2.9.1`
- `torchvision==0.24.1` 
- `transformers==5.9.0`
- `pillow==12.0.0`

## Installation Steps

### Method 1: Quick Installation (Recommended)
```bash
# Activate the dinov3 environment (if not already active)
conda activate dinov3

# Install additional dependencies for SigLIP-2
pip install requests accelerate

# Verify installation
python -c "from transformers import AutoModel, AutoProcessor; print('SigLIP-2 ready!')"
```

### Method 2: Add to Environment Configuration
If you want to make this permanent in your conda environment:

1. **Edit conda.yaml**:
```bash
# Add to the pip section in conda.yaml:
- requests
- accelerate
```

2. **Update environment**:
```bash
conda env update -f conda.yaml
```

## Basic Usage Example

### 1. Simple Text-Image Matching
```python
import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import requests

# Load model and processor
model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

# Load image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Define text queries
texts = ["a cat", "a dog", "a bird", "furniture"]

# Process inputs
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

# Compute similarity scores
similarities = torch.cosine_similarity(text_embeds, image_embeds.unsqueeze(0), dim=-1)
print("Similarity scores:", similarities)
```

### 2. Batch Processing
```python
# For multiple images
images = [image1, image2, image3]  # List of PIL Images
texts = ["description 1", "description 2"]

inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
outputs = model(**inputs)
```

### 3. Using Different Model Variants
```python
# Available models:
models = [
    "google/siglip2-base-patch16-224",     # Base model
    "google/siglip2-so400m-patch14-384",   # Large model
    # Add more variants as available
]

# Load specific model
model_name = "google/siglip2-so400m-patch14-384"
model = AutoModel.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
```

## Advanced Configuration

### 1. Memory Optimization
```python
from transformers import BitsAndBytesConfig

# 4-bit quantization for memory efficiency
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModel.from_pretrained(
    "google/siglip2-base-patch16-224", 
    quantization_config=bnb_config, 
    device_map="auto"
)
```

### 2. Custom Tokenizer (for text-only tasks)
```python
from transformers import Siglip2Tokenizer

tokenizer = Siglip2Tokenizer.from_pretrained("google/siglip2-so400m-patch14-384")
inputs = tokenizer(
    ["HELLO WORLD"], 
    padding="max_length", 
    truncation=True, 
    max_length=64, 
    return_tensors="pt"
)
```

### 3. Integration with DINOv3
```python
# Example: Combine SigLIP-2 with DINOv3 features
import torch.nn.functional as F

# Get DINOv3 features (assuming you have dinov3 loaded)
# dinov3_features = your_dinov3_model(image)

# Get SigLIP-2 features
siglip2_inputs = processor(images=image, return_tensors="pt")
siglip2_features = model.get_image_features(**siglip2_inputs)

# Combine features (example)
# combined_features = torch.cat([dinov3_features, siglip2_features], dim=-1)
```

## Verification Script
Create a test file to verify everything works:

```python
# test_siglip2.py
import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import requests

def test_siglip2():
    try:
        # Load model
        model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
        
        # Test with sample image
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        texts = ["a nature path", "a cityscape"]
        
        # Process
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        
        print("✅ SigLIP-2 installation successful!")
        print(f"Image embeddings shape: {outputs.image_embeds.shape}")
        print(f"Text embeddings shape: {outputs.text_embeds.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_siglip2()
```

## Troubleshooting

### Common Issues:

1. **Import Error**: 
   ```bash
   pip install --upgrade transformers
   ```

2. **CUDA/Memory Issues**:
   ```python
   # Use CPU if GPU memory is insufficient
   model = model.to('cpu')
   ```

3. **Version Conflicts**:
   ```bash
   pip install transformers>=4.35.0
   ```

### System Requirements:
- **RAM**: Minimum 8GB (16GB+ recommended for large models)
- **Storage**: ~2-5GB for model weights
- **GPU**: Optional but recommended for faster inference

## Integration with Your Current Workflow

Since you're working in a DINOv3 environment, you can:

1. **Use SigLIP-2 for text-image retrieval** alongside DINOv3's dense features
2. **Combine embeddings** for multimodal tasks
3. **Create hybrid pipelines** using both models' strengths

## Next Steps

1. Run the verification script
2. Test with your specific use case
3. Consider model fine-tuning if needed
4. Explore multimodal combinations with DINOv3

---
*Generated for DINOv3 environment on 2026-06-10*