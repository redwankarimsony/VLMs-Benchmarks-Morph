import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
from typing import List
from tqdm import tqdm
from dataset import ImageDataset



# Find the number of GPUs available
num_gpus = torch.cuda.device_count()

# Device
device = torch.device(f"cuda:{num_gpus-1}" if torch.cuda.is_available() else "cpu")

# Model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)

# Freeze vision encoder
for param in model.vision_model.parameters():
    param.requires_grad = False


# Collate function for batch preprocessing
def collate_fn(batch):
    return processor(images=batch, return_tensors="pt", padding=True)


# Feature extraction function
def extract_clip_features(image_paths: List[str], batch_size=128, use_cls=True):
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=16, pin_memory=True, shuffle=False)

    all_features = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = {k: v.to(device) for k, v in batch.items()}
            vision_outputs = model.vision_model(**batch)
            hidden_states = vision_outputs.last_hidden_state  # [B, N, D]

            if use_cls:
                features = hidden_states[:, 0, :]  # CLS token
            else:
                features = hidden_states[:, 1:, :].mean(dim=1)  # Mean of patch embeddings

            all_features.append(features.cpu())
            

    return torch.cat(all_features, dim=0)  # shape: [num_images, dim]


def load_image_paths_from_file(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    return image_paths

# 🔧 Example usage:
if __name__ == "__main__":
    # Example image folder
    AMSL_PATH = "extracted_features/AMSL.txt"
    FACELAB_LONDON_PATH = "extracted_features/facelab_london.txt"
    SMDD_PATH = "extracted_features/SMDD.txt"
    MORDIFF_PATH = "extracted_features/fraunhofer_MorDiff.txt"
    
    
    
    # Load image paths from text files
    amsl_image_paths = load_image_paths_from_file(AMSL_PATH)
    facelab_london_image_paths = load_image_paths_from_file(FACELAB_LONDON_PATH)
    smdd_image_paths = load_image_paths_from_file(SMDD_PATH)
    mor_diff_image_paths = load_image_paths_from_file(MORDIFF_PATH)
    
    # Extract features
    amsl_features = extract_clip_features(amsl_image_paths, batch_size=128)
    facelab_london_features = extract_clip_features(facelab_london_image_paths, batch_size=128)
    smdd_features = extract_clip_features(smdd_image_paths, batch_size=128)
    mor_diff_features = extract_clip_features(mor_diff_image_paths, batch_size=128)
    
   # Save CLIP features
    save_dir = os.path.join("extracted_features", "CLIP")
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(amsl_features, os.path.join(save_dir, "amsl_features.pt"))
    torch.save(facelab_london_features, os.path.join(save_dir, "facelab_london_features.pt"))
    torch.save(smdd_features, os.path.join(save_dir, "smdd_features.pt"))
    torch.save(mor_diff_features, os.path.join(save_dir, "mor_diff_features.pt"))
    print("Features extracted and saved successfully.")
    