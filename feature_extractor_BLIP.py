# extract_blip_features.py

import os
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import BlipProcessor, BlipForConditionalGeneration

# ======================
# CONFIG
# ======================
blip_model_name = "Salesforce/blip-image-captioning-base"
input_size = 224
batch_size = 64
num_workers = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_PATHS = {
    "amsl": "extracted_features/AMSL.txt",
    "facelab_london": "extracted_features/facelab_london.txt",
    "smdd": "extracted_features/SMDD.txt",
    "mordiff": "extracted_features/fraunhofer_MorDiff.txt"
}

OUTPUT_DIR = f"extracted_features/{blip_model_name.split('/')[-1]}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# TRANSFORM
# ======================
def build_blip_transform(input_size=224):
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # BLIP expects [-1, 1]
    ])

# ======================
# DATASET
# ======================
class BLIPImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.image_paths[idx]

# ======================
# UTILITY
# ======================
def load_image_paths_from_file(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.read().splitlines()
    return [line.strip() for line in lines if line.strip()]

# ======================
# FEATURE EXTRACTOR
# ======================
def extract_blip_features_from_dataloader(vision_model, dataloader, device):
    all_features = []
    all_filenames = []

    vision_model.eval()
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Extracting BLIP Features"):
            pixel_values = images.to(device)
            outputs = vision_model(pixel_values)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            all_features.append(cls_embeddings.cpu())
            all_filenames.extend(paths)

    all_features = torch.cat(all_features, dim=0)
    return all_features, all_filenames

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    # Load model
    processor = BlipProcessor.from_pretrained(blip_model_name)
    full_model = BlipForConditionalGeneration.from_pretrained(blip_model_name).to(device).eval()
    vision_model = full_model.vision_model

    transform = build_blip_transform(input_size=input_size)

    for dataset_name, txt_path in DATA_PATHS.items():
        print(f"\nProcessing dataset: {dataset_name.upper()}")

        image_paths = load_image_paths_from_file(txt_path)

        dataset = BLIPImageDataset(image_paths, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                pin_memory=True, shuffle=False)

        features, filenames = extract_blip_features_from_dataloader(vision_model, dataloader, device)

        # Save features
        out_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_features.pt")
        torch.save(features, out_path)
        print(f"Saved {features.shape[0]} feature vectors to {out_path}")