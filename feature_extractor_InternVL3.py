#%%
import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
from feature_extractor_CLIP import load_image_paths_from_file
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os
vision_model_idx = 0


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    model_path = "/mnt/scratch/sonymd/.cache/huggingface/hub/models--OpenGVLab--InternVL3-78B/snapshots/9e3847ed2bcc7a31535af8df019a7537f22a2078"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    
    device_map['vision_model'] = vision_model_idx
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


#%%
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = build_transform(input_size=448)
        self.image_size = 448
        self.max_num = 12
        self.use_thumbnail = True
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        pixel_values = Image.open(self.image_paths[idx]).convert('RGB')
        pixel_values = self.transform(pixel_values)
        return pixel_values, self.image_paths[idx]
    




#%%
def extract_features_from_dataloader(vision_model, dataloader, device=torch.device(f'cuda:{vision_model_idx}')):
    all_features = []
    all_filenames = []

    vision_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            images , paths = batch
            pixel_values = images.to(torch.bfloat16).to(device) 
            
            # Forward pass
            outputs = vision_model(pixel_values)
            features = outputs.last_hidden_state[:, 0, :]  # CLS token

            all_features.append(features.cpu())
            all_filenames.extend(paths)

    # Stack all features into a single tensor
    all_features = torch.cat(all_features, dim=0)
    return all_features



#%% Load the model
# Load model
path = 'OpenGVLab/InternVL3-78B'
device_map = split_model('InternVL3-78B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
model = model.vision_model



#%% Create and Execute the Dataset and DataLoader
AMSL_PATH = "extracted_features/AMSL.txt"
FACELAB_LONDON_PATH = "extracted_features/facelab_london.txt"
SMDD_PATH = "extracted_features/SMDD.txt"
FRAUNHOFER_MORDIFF_PATH = "extracted_features/fraunhofer_MorDiff.txt"


# Load image paths from text files
amsl_image_paths = load_image_paths_from_file(AMSL_PATH)
facelab_london_image_paths = load_image_paths_from_file(FACELAB_LONDON_PATH)
smdd_image_paths = load_image_paths_from_file(SMDD_PATH)
mordiff_image_paths = load_image_paths_from_file(FRAUNHOFER_MORDIFF_PATH)

smdd_image_paths = smdd_image_paths[100000:130000]  # Limit to 50k images

ds_amsl = ImageDataset(amsl_image_paths, transform=build_transform(input_size=448))
ds_facelab_london = ImageDataset(facelab_london_image_paths, transform=build_transform(input_size=448))
ds_smdd = ImageDataset(smdd_image_paths, transform=build_transform(input_size=448))
ds_mordiff = ImageDataset(mordiff_image_paths, transform=build_transform(input_size=448))



# Extract the AMSL Features 
dl_amsl = DataLoader(ds_amsl, batch_size=128, num_workers=16, pin_memory=True, shuffle=False)
dl_facelab_london = DataLoader(ds_facelab_london, batch_size=128, num_workers=16, pin_memory=True, shuffle=False)
dl_smdd = DataLoader(ds_smdd, batch_size=128, num_workers=16, pin_memory=True, shuffle=False)
dl_mordiff = DataLoader(ds_mordiff, batch_size=128, num_workers=16, pin_memory=True, shuffle=False)

output_dir = "extracted_features/InternVL3"
os.makedirs(output_dir, exist_ok=True)


# #%% Extract features for AMSL
# features_amsl = extract_features_from_dataloader(vision_model, dl_amsl)
# torch.save(features_amsl, os.path.join(output_dir, "amsl_features.pt"))


# #%% Extract features for Facelab London
# features_facelab_london = extract_features_from_dataloader(vision_model, dl_facelab_london)
# torch.save(features_facelab_london, os.path.join(output_dir, "facelab_london_features.pt"))



# #%% Extract features for MorDiff
# features_mordiff = extract_features_from_dataloader(vision_model, dl_mordiff)
# torch.save(features_mordiff, os.path.join(output_dir, "mordiff_features.pt"))


#%% Extract features for SMDD
features_smdd = extract_features_from_dataloader(model, dl_smdd)
torch.save(features_smdd, os.path.join(output_dir, "smdd_features_100000_130000.pt"))
