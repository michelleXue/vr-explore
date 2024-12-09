import os
from pydantic import BaseModel
from typing import List
from utils import encode_image

class DetectedObject(BaseModel):
    object_name: str
    left_top_coordinate_x: float
    left_top_coordinate_y: float
    right_bottom_coordinate_x: float
    right_bottom_coordinate_y: float

class SceneCoordinateAnalysis(BaseModel):
    objects: List[DetectedObject]

DETECTION_PROMPT = """
Your task is to analyze the provided image and return structured data. Follow these rules strictly:

1. Scene:
   - List every object individually. 

2. For each object, provide:
   - Name: The object name with a unique identifier (e.g., 'Chair A', 'Chair B').
   - Coordinates:
     - left_top_coordinate_x
     - left_top_coordinate_y
     - right_bottom_coordinate_x
     - right_bottom_coordinate_y
"""

VERIFICATION_PROMPT = """
Examine the image carefully to see if there are any missing items. If there are, add them and return the complete JSON.
"""

def detect_objects_coordinate(image_path, call_api_func, prompts=[DETECTION_PROMPT, VERIFICATION_PROMPT]):
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
                            "detail": "low"
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
        
        result = call_api_func(messages, SceneCoordinateAnalysis)
    
    # Save result
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    with open(f"{image_filename}.json", 'w', encoding='utf-8') as f:
        f.write(result)
    
    return result 