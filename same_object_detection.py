import os
from pydantic import BaseModel
from utils import encode_image

class ComparisonResult(BaseModel):
    is_same_object: bool
    confidence_score: float
    object1_description: str
    object2_description: str
    reasoning: str

COMPARISON_PROMPT = """
Your task is to analyze two images and determine if the objects in the specified regions are the same object.
For Image 1, focus on the object within the {box1_color} box.
For Image 2, focus on the object within the {box2_color} box.

Please provide:
1. Whether they are the same object (true/false)
2. Your confidence score (0.0 to 1.0)
3. Description of object in Image 1
4. Description of object in Image 2
5. Detailed reasoning for your conclusion, comparing:
   - Object name and type
   - Color
   - Shape
   - Material
   - Texture
   - Brightness
   - Orientation
   - Size and proportions
   - Any distinctive features or unique identifying marks
"""

VERIFICATION_PROMPT = """
Please review your comparison carefully. Consider:
1. Are there any subtle details you might have missed?
2. Could any object differences affect your judgment?
3. Are there any unique identifying features that definitively prove or disprove it's the same object?

Provide an updated analysis if needed.
"""

def compare_objects(image1_path, image2_path, call_api_func, box1_color, box2_color, prompts):
    """Compare objects in specified regions of two images"""
    base64_image1 = encode_image(image1_path)
    base64_image2 = encode_image(image2_path)
    messages = []
    result = None
    
    for i, current_prompt in enumerate(prompts):
        if i == 0:
            formatted_prompt = current_prompt.format(box1_color=box1_color, box2_color=box2_color)
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image1}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image2}",
                            "detail": "high"
                        }
                    },
                    {"type": "text", "text": formatted_prompt}
                ]
            })
        else:
            messages.extend([
                {"role": "assistant", "content": [{"type": "text", "text": result}]},
                {"role": "user", "content": [{"type": "text", "text": current_prompt}]}
            ])
        
        result = call_api_func(messages, ComparisonResult)
    
    # Save result
    image1_filename = os.path.splitext(os.path.basename(image1_path))[0]
    image2_filename = os.path.splitext(os.path.basename(image2_path))[0]
    result_filename = f"comparison_{image1_filename}_{image2_filename}_{box1_color}-{box2_color}.json"
    
    with open(result_filename, 'w', encoding='utf-8') as f:
        f.write(result)
    
    return result 