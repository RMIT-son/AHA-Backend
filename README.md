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

# === Download OpenAI CLIP RN50 ===
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
