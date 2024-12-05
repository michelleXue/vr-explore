import base64
from openai import OpenAI
from pydantic import BaseModel
from typing import List
import os
import cv2

short_prompt = """
What is in the picture?
"""

prompt = """
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

prompt_2 = """
Examine the image carefully to see if there are any missing items. If there are, add them and return the complete JSON.
"""

class DetectedObject(BaseModel):
    object_name: str  # Object name with unique identifier (e.g., "Chair A", "Chair B")
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

def encode_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
        
    # Encode the image
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def analyze_image(image_path):
    client = OpenAI(
        api_key="",
        base_url=""
    )
    
    messages = []
    base64_image = encode_image(image_path)
    
    def get_completion(messages):
        return client.beta.chat.completions.parse(
            model="chatgpt-4o-latest",
            messages=messages,
            response_format=SceneAnalysis
        ).choices[0].message.content
    
    result = None
    
    # Process each prompt in sequence
    for i, current_prompt in enumerate(prompts):
        if i == 0:
            # First prompt always includes the image
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
            # Subsequent prompts build on previous results
            messages.extend([
                {"role": "assistant", "content": [{"type": "text", "text": result}]},
                {"role": "user", "content": [{"type": "text", "text": current_prompt}]}
            ])
        
        result = get_completion(messages)
    
    # Save final result
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    with open(f"{image_filename}.json", 'w', encoding='utf-8') as f:
        f.write(result)
    
    return result

def process_images_in_folder(folder_path):
    # List all files in the folder
    for filename in os.listdir(folder_path):
        # Construct full file path
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's an image file
        if filename.lower().endswith(('_1.png', '_6.png', '_11.png')):
            # Analyze the image
            result = analyze_image(file_path, prompts=[prompt, prompt_2, prompt_2])
            
if __name__ == "__main__":
    folder_path = "dataset"
    process_images_in_folder(folder_path)
