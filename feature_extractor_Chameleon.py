# extract_chameleon_features.py

import os
from PIL import Image
from tqdm import tqdm
import torch
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
import argparse
import psutil
import os



# ======================
# CONFIG
# ======================
chameleon_model_name = "facebook/chameleon-7b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATHS = {
    # "amsl": "extracted_features/AMSL.txt",
    # "facelab_london": "extracted_features/facelab_london.txt",
    # "mordiff": "extracted_features/fraunhofer_MorDiff.txt",
    "smdd": "extracted_features/SMDD.txt"

}

OUTPUT_DIR = f"extracted_features/{chameleon_model_name.split('/')[-1]}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# UTILITY
# ======================
def load_image_paths_from_file(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]
# ======================
# MAIN
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features using Chameleon model")
    parser.add_argument('--start', type=int, default=0, help='Start index for processing images')
    parser.add_argument('--end', type=int, default=100000, help='End index for processing images')
    args = parser.parse_args()
    # Load model
    processor = ChameleonProcessor.from_pretrained(chameleon_model_name)
    model = ChameleonForConditionalGeneration.from_pretrained(
        chameleon_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    model.eval()

    batch_size = 16  # Try 8, adjust based on VRAM

    for dataset_name, txt_path in DATA_PATHS.items():
        print(f"\nProcessing dataset: {dataset_name.upper()}")
        image_paths = load_image_paths_from_file(txt_path)
        
        image_paths = image_paths[args.start:args.end]  # Limit to 50k images

        all_features = []
        all_filenames = []

        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []

            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert("RGB")
                    batch_images.append(image)
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")
                    continue

            if not batch_images:
                continue

            try:
                inputs = processor(images=batch_images, text=["<image>"] * len(batch_images),
                                   return_tensors="pt", padding=True).to(device, torch.bfloat16)

                with torch.no_grad():
                    outputs = model.model(**inputs, output_hidden_states=True)
                    cls_feats = outputs.hidden_states[-1][:, 0, :]  # CLS token
                    all_features.append(cls_feats.cpu())
                    all_filenames.extend(batch_paths[:len(cls_feats)])

            except Exception as e:
                print(f"Error in batch starting at {i}: {e}")
                continue

        if all_features:
            features_tensor = torch.cat(all_features, dim=0)
            out_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_features_{args.start}_{args.end}.pt")
            torch.save(features_tensor, out_path)
            print(f"Saved {features_tensor.shape[0]} feature vectors to {out_path}")