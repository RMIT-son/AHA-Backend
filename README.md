# AHA-Backend
Backend for AHA system instruction for testing
1. Clone repo, cd to repo and run **python -m venv venv** to create virtual enviroment, run **venv\Scripts\activate** to activate
2. Run **pip install -r requirements.txt** in virtual enviroment to install all the required packages 
3. Create a **python file** copy and run the code below to install all the required models:
4. Run command **uvicorn app.main:app** to run FastAPI
5. Go to **localhost:8000/docs#/** to test the model

**Note:** check last 2 lines in the requirements.txt if you cannot install requirements.txt, you will have to install that last 2 packages separately so just command those 2 line temporary and run pip install -r requirements.txt again if it fails to install, and then run pip install those 2 packages later

**PLEASE COPY THIS CODE TO A PYTHON FILE AND RUN IT**
import os
import requests
from huggingface_hub import snapshot_download

# === Part 1: Download OpenAI CLIP RN50 ===
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

# === Part 2: Download Hugging Face Models ===
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
