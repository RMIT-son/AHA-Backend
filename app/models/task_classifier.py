import os
import base64
import asyncio
import requests
from PIL import Image
from io import BytesIO
from typing import Optional, Union
from transformers import pipeline
from .zero_shot_image_classifier import ZeroShotImageClassification

class Classifier:
    """Classifier for text, general images, and disease-related images using zero-shot models."""

    def __init__(self, config: dict = None):
        self.config = config
        self.candidate_labels = self.config["candidate_labels"]
        self.zero_shot_text_classification = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli"
        )
        self.zero_shot_image_classification = ZeroShotImageClassification()
        self.zero_shot_disease_classification = pipeline(
            "image-classification",
            model="krzonkalla/Detector_de_Cancer_de_Pele"
        )

    async def classify_text(self, prompt: str = None) -> str:
        """
        Classify a text prompt using zero-shot classification.

        This function uses a zero-shot text classification model to determine the most appropriate
        label from a predefined list (`self.candidate_labels`), such as ["not-medical-related", "dermatology",...].

        Args:
            prompt (str, optional): The input text to be classified.

        Returns:
            str: The top predicted label based on the input text.
        """
        result = await asyncio.to_thread(
            self.zero_shot_text_classification,
            prompt,
            candidate_labels=self.candidate_labels
        )
        return result["labels"][0]


    def _load_image(self, image: Union[str, Image.Image] = None) -> Image.Image:
        """
        Helper function to load and standardize image input.

        Supports loading from local file paths, URLs, or PIL.Image/Image-like objects.

        Args:
            image (str | Image.Image, optional): The image input, which can be a file path, URL, or PIL Image object.

        Returns:
            Image.Image: A loaded and RGB-converted PIL image.

        Raises:
            ValueError: If the input is invalid or unsupported.
        """
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                response = requests.get(image)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")

            elif image.startswith("data:image/"):
                # Handle base64-encoded image string
                try:
                    header, base64_data = image.split(",", 1)
                    decoded_bytes = base64.b64decode(base64_data)
                    return Image.open(BytesIO(decoded_bytes)).convert("RGB")
                except Exception as e:
                    raise ValueError(f"Failed to decode base64 image: {e}")

            elif os.path.isfile(image):
                return Image.open(image).convert("RGB")

            else:
                raise ValueError("Invalid image path, URL, or base64 string.")

        if isinstance(image, Image.Image):
            return image.to_pil().convert("RGB") if hasattr(image, "to_pil") else image.convert("RGB")

        raise ValueError("Unsupported image input type.")


    async def classify_image(self, image: Optional[str | Image.Image] = None) -> str:
        """
        Classify a general image using a zero-shot image classification model.

        This is useful for general-purpose tasks such as visual content detection, dermatology pre-screening, etc.

        Args:
            image (str | Image.Image, optional): The input image (local path, URL, or PIL image).

        Returns:
            str: The top predicted label from the predefined candidate labels.
        """
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
        """
        Classify a medical condition from an image using a fine-tuned disease classification model.

        Designed for healthcare/dermatology applications where accurate disease labeling is required.

        Args:
            image (str | Image.Image, optional): The medical image to classify (e.g., skin lesion image).

        Returns:
            str: A detailed classification message including the predicted disease and a standard note.

        Example:
            "Response from the disease image classifier: The provided image may show signs of psoriasis, please consult a healthcare professional."
        """
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
