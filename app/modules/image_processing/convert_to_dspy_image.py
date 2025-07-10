import os
import io
import base64
import requests
import tempfile
from dspy import Image
from pathlib import Path
from PIL import Image as PILImage
from typing import Union

def convert_to_dspy_image(image_data: Union[str, bytes, PILImage.Image, io.BytesIO] = None) -> Image:
    """
    Convert various image data types to dspy Image
    
    Args:
        image_data: Can be URL, file path, base64, bytes, PIL Image, or BytesIO
    
    Returns:
        dspy.Image object
    """
    # Use temporary file with proper cleanup
    temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
    
    try:
        # First, get the image as PIL Image for processing
        pil_image = _convert_to_pil(image_data)
        
        # Close the file descriptor before saving
        if temp_fd is not None:
            os.close(temp_fd)
            temp_fd = None
            
        # Save image to temp file
        pil_image.save(temp_path, format='JPEG')
        
        # Load with dspy
        dspy_image = Image.from_file(temp_path)
        
        return dspy_image
        
    except Exception as e:
        raise e
        
    finally:
        # Clean up temporary file and file descriptor
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except:
                pass
        
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def _convert_to_pil(image_data: Union[str, bytes, PILImage.Image, io.BytesIO] = None) -> PILImage.Image:
    """Convert input data to PIL Image"""
    if isinstance(image_data, str):
        if image_data.startswith(('http://', 'https://')):
            # URL
            response = requests.get(image_data, timeout=30)
            response.raise_for_status()
            return PILImage.open(io.BytesIO(response.content))
            
        elif image_data.startswith('data:image'):
            # Base64 with data URI prefix
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            return PILImage.open(io.BytesIO(image_bytes))
            
        elif _is_base64(image_data):
            # Plain base64 string
            try:
                image_bytes = base64.b64decode(image_data)
                return PILImage.open(io.BytesIO(image_bytes))
            except Exception:
                # If base64 decode fails, treat as file path
                return _handle_file_path_pil(image_data)
                
        else:
            # Local file path
            return _handle_file_path_pil(image_data)
            
    elif isinstance(image_data, bytes):
        return PILImage.open(io.BytesIO(image_data))
        
    elif isinstance(image_data, PILImage.Image):
        return image_data.copy()
        
    elif isinstance(image_data, io.BytesIO):
        image_data.seek(0)
        return PILImage.open(image_data)
        
    else:
        raise ValueError(f"Unsupported image data type: {type(image_data)}")


def _is_base64(string: str = None) -> bool:
    """Check if string is valid base64"""
    try:
        if len(string) % 4 != 0:
            return False
        base64.b64decode(string, validate=True)
        return True
    except Exception:
        return False


def _handle_file_path_pil(file_path: str = None) -> PILImage.Image:
    """Handle local file path conversion to PIL"""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    if path.suffix.lower() not in valid_extensions:
        raise ValueError(f"File does not appear to be an image: {file_path}")
    
    return PILImage.open(str(path))