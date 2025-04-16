# extract_blip2_features.py

import os
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import Blip2Processor, Blip2Model

# ======================
# CONFIG
# ======================
blip2_model_name = "Salesforce/blip2-opt-2.7b"
blip2_model_name = "Salesforce/blip2-opt-6.7b"
blip2_model_name = "Salesforce/blip2-flan-t5-xl-coco"
# blip2_model_name = "Salesforce/blip2-flan-t5-xl"
# blip2_model_name = "Salesforce/blip2-flan-t5-xxl"

input_size = 224  # default for BLIP-2
batch_size = 64
num_workers = 8
device_count = torch.cuda.device_count()
device = torch.device(f"cuda:{device_count-1}" if torch.cuda.is_available() else "cpu")

DATA_PATHS = {
    "amsl": "extracted_features/AMSL.txt",
    "facelab_london": "extracted_features/facelab_london.txt",
    "smdd": "extracted_features/SMDD.txt",
    "mordiff": "extracted_features/fraunhofer_MorDiff.txt"
}

OUTPUT_DIR = f"extracted_features/{blip2_model_name.split('/')[-1]}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# TRANSFORM
# ======================
def build_blip2_transform(input_size=224):
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # BLIP-2 expects [-1, 1]
    ])

# ======================
# DATASET
# ======================
class BLIP2ImageDataset(Dataset):
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
# UTIL
# ======================
def load_image_paths_from_file(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.read().splitlines()
    return [line.strip() for line in lines if line.strip()]

# ======================
# FEATURE EXTRACTION
# ======================
def extract_blip2_qformer_features(model, dataloader, processor, device):
    all_features, all_filenames = [], []

    model.eval()
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Extracting BLIP-2 Q-Former Features"):
            # Use processor to preprocess images for BLIP-2
            inputs = processor(images=[T.ToPILImage()(img) for img in images], return_tensors="pt").to(device)
            qformer_outputs = model.get_qformer_features(**inputs)  # (B, num_query_tokens, hidden_dim)
            qformer_outputs = qformer_outputs.last_hidden_state  # (B, num_query_tokens, hidden_dim)
            pooled_features = qformer_outputs.mean(dim=1)           # (B, hidden_dim)

            all_features.append(pooled_features.cpu())
            all_filenames.extend(paths)

    all_features = torch.cat(all_features, dim=0)
    return all_features, all_filenames

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    print(f"Loading model {blip2_model_name}...")
    processor = Blip2Processor.from_pretrained(blip2_model_name)
    model = Blip2Model.from_pretrained(blip2_model_name, torch_dtype=torch.float16).to(device).eval()

    transform = build_blip2_transform(input_size=input_size)

    for dataset_name, txt_path in DATA_PATHS.items():
        print(f"\nProcessing dataset: {dataset_name.upper()}")

        image_paths = load_image_paths_from_file(txt_path)


        dataset = BLIP2ImageDataset(image_paths, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                pin_memory=True, shuffle=False)

        features, filenames = extract_blip2_qformer_features(model, dataloader, processor, device)

        out_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_qformer_features.pt")
        torch.save(features, out_path)
        print(f"Saved {features.shape[0]} feature vectors to {out_path}")