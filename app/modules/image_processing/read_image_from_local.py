from dspy import Image

def read_image_from_local(file_path: str = None) -> Image:
    """Load image from local path"""
    try:
        image = Image.from_file(file_path)
        return image
        
    except Exception as e:
        print(f"Error processing local image: {e}")