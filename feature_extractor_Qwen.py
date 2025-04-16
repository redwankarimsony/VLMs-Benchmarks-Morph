#%% 
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
from qwen_vl_utils import process_vision_info

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
).eval()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Load image
image = Image.open("/mnt/home/sonymd/research/cropped_datasets/SMDD/os25k_m_t/img554510.png").convert("RGB")

# Build dummy message to use processor & vision pipeline
messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]



# Preprocess input
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
image_inputs, _ = process_vision_info(messages)



inputs = processor(
    text=[text],
    images=image_inputs,
    return_tensors="pt",
    padding=True
).to(model.device)


#%%
# Extract image features (vision encoder output)
with torch.no_grad():
    vision_outputs = model.visual(image_inputs[0].unsqueeze(0))

# CLS token feature vector (usually at position 0)
image_feature = vision_outputs.last_hidden_state[:, 0, :]  # Shape: [1, hidden_dim]

print("Image feature vector shape:", image_feature.shape)
