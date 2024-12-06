import os
from pydantic import BaseModel
from typing import List
from utils import encode_image

class DetectedObject(BaseModel):
    object_name: str
    color: str
    orientation: str
    location: str
    brightness: str
    shape: str
    texture: str
    material: str
    relationship_with_user_horizontal: str
    relationship_with_user_vertical: str
    relationship_with_user_distance: str

class SceneAnalysis(BaseModel):
    objects: List[DetectedObject]
    scene_type: str

DETECTION_PROMPT = """
Your task is to analyze the provided image and return structured data. Follow these rules strictly:

1. Scene Type:
   - Provide the type of scene (e.g., dining room, office).
   - List every object individually. 

2. For each object, provide:
   - Name: The object name with a unique identifier (e.g., 'Chair A', 'Chair B').
   - Color
   - Orientation: facing direction
   - Location: in the scene
   - Brightness
   - Shape
   - Texture
   - Material
   - Spatial Relationship:
     - Horizontal: Specify left, middle, or right.
     - Vertical: Specify up, down, or center.
     - Distance: Specify near, mid-range, or far.
"""

VERIFICATION_PROMPT = """
Examine the image carefully to see if there are any missing items. If there are, add them and return the complete JSON.
"""

def detect_objects(image_path, call_api_func, prompts=[DETECTION_PROMPT]):
    """Detect and analyze objects in the image"""
    base64_image = encode_image(image_path)
    messages = []
    result = None
    
    for i, current_prompt in enumerate(prompts):
        if i == 0:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    },
                    {"type": "text", "text": current_prompt}
                ]
            })
        else:
            messages.extend([
                {"role": "assistant", "content": [{"type": "text", "text": result}]},
                {"role": "user", "content": [{"type": "text", "text": current_prompt}]}
            ])
        
        result = call_api_func(messages, SceneAnalysis)
    
    # Save result
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    with open(f"{image_filename}.json", 'w', encoding='utf-8') as f:
        f.write(result)
    
    return result 