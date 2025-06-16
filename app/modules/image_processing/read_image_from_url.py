from dspy import Image

def read_image_from_url(url: str = None) -> Image:
    """Alternative method: Load image directly from URL with DSPy"""
    try:
        # DSPy Image can sometimes load directly from URL
        image = Image.from_url(url)
        return image
        
    except Exception as e:
        print(f"Error loading image directly from URL: {e}")