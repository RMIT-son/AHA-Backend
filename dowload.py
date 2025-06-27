import os
import requests
from huggingface_hub import snapshot_download

clip_url = "https://openaipublic.blob.core.windows.net/clip/models/RN50.pt"
clip_dest_dir = "zero_shot_image_classification"
clip_dest_file = os.path.join(clip_dest_dir, "RN50.pt")

os.makedirs(clip_dest_dir, exist_ok=True)

if not os.path.exists(clip_dest_file):
    print(f"Downloading OpenAI CLIP RN50 to {clip_dest_file} ...")
    response = requests.get(clip_url)
    with open(clip_dest_file, 'wb') as f:
        f.write(response.content)
    print("Download complete!")
else:
    print("CLIP RN50 already downloaded.")
    
    
hf_target_dir = "huggingface_models"
os.makedirs(hf_target_dir, exist_ok=True)

models = {
    "intfloat/multilingual-e5-small": "multilingual-e5-small",
    "naver/splade-cocondenser-ensembledistil": "splade-cocondenser",
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli": "deb-v3-large-mnli",
    "krzonkalla/Detector_de_Cancer_de_Pele": "cancer-detector"
}

for model_id, folder_name in models.items():
    model_path = os.path.join(hf_target_dir, folder_name)
    if not os.path.exists(model_path) or not os.listdir(model_path):
        print(f"Downloading {model_id} to {model_path} ...")
        snapshot_download(
            repo_id=model_id,
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        print(f"{model_id} downloaded.")
    else:
        print(f"{model_id} already downloaded.")
