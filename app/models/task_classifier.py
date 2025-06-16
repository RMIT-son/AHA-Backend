import os
import asyncio
import requests
from PIL import Image
from io import BytesIO
from typing import Optional
from transformers import pipeline
from ZSIC import ZeroShotImageClassification

class Classifier:
    """Classifier for text, general images, and disease-related images using zero-shot models."""

    def __init__(self, config: dict = None):
        self.config = config
        self.candidate_labels = self.config["candidate_labels"]
        self.zero_shot_text_classification = pipeline(
            "zero-shot-classification", 
            model=self.config["model"]
        )
        self.zero_shot_image_classification = ZeroShotImageClassification()
        self.zero_shot_disease_classification = pipeline(
            "image-classification",
            model="huggingface_models/cancer-detector"
        )

    async def classify_text(self, prompt: str = None) -> str:
        """Classify a text prompt using zero-shot classification."""
        result = await asyncio.to_thread(
            self.zero_shot_text_classification,
            prompt,
            candidate_labels=self.candidate_labels
        )
        return result["labels"][0]

    def _load_image(self, image: str | Image.Image = None) -> Image.Image:
        """Helper function to load image from various formats."""

        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                response = requests.get(image)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            elif os.path.isfile(image):
                return Image.open(image).convert("RGB")
            else:
                raise ValueError("Invalid image path or URL.")

        if isinstance(image, Image):
            return image.to_pil().convert("RGB") if hasattr(image, "to_pil") else image

        raise ValueError("Unsupported image input type.")

    async def classify_image(self, image: Optional[str | Image.Image] = None) -> str:
        """Classify general image (e.g., dermatology task) using custom ZSIC."""
        try:
            img_data = self._load_image(image) if image else None
            result = await asyncio.to_thread(
                self.zero_shot_image_classification,
                image=img_data,
                candidate_labels=self.candidate_labels
            )
            return result["labels"][0]

        except Exception as e:
            print(f"[Image Classification] Error: {e}")
            return "classification_failed"

    async def classify_disease(self, image: Optional[str | Image.Image] = None) -> str:
        """Classify a medical condition from an image using fine-tuned HF model."""
        try:
            img_data = self._load_image(image) if image else None
            result = await asyncio.to_thread(
                self.zero_shot_disease_classification,
                img_data
            )
            note = self.config["disease_response_note"]
            label = result[0]["label"]
            return f"Response from the disease image classifier: The provided image may show signs of {label}, {note}"

        except Exception as e:
            print(f"[Disease Classification] Error: {e}")
            return "classification_failed"