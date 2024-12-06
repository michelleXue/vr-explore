import os
from openai import OpenAI
from object_detection import detect_objects, DETECTION_PROMPT, VERIFICATION_PROMPT
from same_object_detection import compare_objects, COMPARISON_PROMPT, VERIFICATION_PROMPT as COMPARISON_VERIFICATION_PROMPT

def call_api(messages, response_model):
    """Make API call to OpenAI"""
    client = OpenAI(
        api_key="",
        base_url=""
    )
    
    return client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=messages,
        response_format=response_model
    ).choices[0].message.content

def object_detection_in_folder(folder_path):
    """Process all images in the specified folder"""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('_1.png', '_6.png', '_11.png')):
            file_path = os.path.join(folder_path, filename)
            result = detect_objects(file_path, call_api, prompts=[DETECTION_PROMPT, VERIFICATION_PROMPT])


if __name__ == "__main__":
    # object_detection_in_folder("dataset")
    compare_objects("dataset-box/scene1_1.png", "dataset-box/scene1_2.png", call_api, box1_color="red", box2_color="blue", prompts=[COMPARISON_PROMPT, COMPARISON_VERIFICATION_PROMPT])
    compare_objects("dataset-box/scene1_1.png", "dataset-box/scene1_2.png", call_api, box1_color="red", box2_color="green", prompts=[COMPARISON_PROMPT, COMPARISON_VERIFICATION_PROMPT])
    compare_objects("dataset-box/scene1_1.png", "dataset-box/scene1_3.png", call_api, box1_color="red", box2_color="blue", prompts=[COMPARISON_PROMPT, COMPARISON_VERIFICATION_PROMPT])
    compare_objects("dataset-box/scene1_1.png", "dataset-box/scene1_3.png", call_api, box1_color="red", box2_color="green", prompts=[COMPARISON_PROMPT, COMPARISON_VERIFICATION_PROMPT])

    compare_objects("dataset-box/scene2_1.png", "dataset-box/scene2_2.png", call_api, box1_color="red", box2_color="blue", prompts=[COMPARISON_PROMPT, COMPARISON_VERIFICATION_PROMPT])
    compare_objects("dataset-box/scene2_1.png", "dataset-box/scene2_2.png", call_api, box1_color="red", box2_color="green", prompts=[COMPARISON_PROMPT, COMPARISON_VERIFICATION_PROMPT])
    compare_objects("dataset-box/scene2_1.png", "dataset-box/scene2_3.png", call_api, box1_color="red", box2_color="blue", prompts=[COMPARISON_PROMPT, COMPARISON_VERIFICATION_PROMPT])
    compare_objects("dataset-box/scene2_1.png", "dataset-box/scene2_3.png", call_api, box1_color="red", box2_color="green", prompts=[COMPARISON_PROMPT, COMPARISON_VERIFICATION_PROMPT])

    compare_objects("dataset-box/scene3_1.png", "dataset-box/scene3_2.png", call_api, box1_color="red", box2_color="blue", prompts=[COMPARISON_PROMPT, COMPARISON_VERIFICATION_PROMPT])
    compare_objects("dataset-box/scene3_1.png", "dataset-box/scene3_2.png", call_api, box1_color="red", box2_color="green", prompts=[COMPARISON_PROMPT, COMPARISON_VERIFICATION_PROMPT])
    compare_objects("dataset-box/scene3_1.png", "dataset-box/scene3_3.png", call_api, box1_color="red", box2_color="blue", prompts=[COMPARISON_PROMPT, COMPARISON_VERIFICATION_PROMPT])
    compare_objects("dataset-box/scene3_1.png", "dataset-box/scene3_3.png", call_api, box1_color="red", box2_color="green", prompts=[COMPARISON_PROMPT, COMPARISON_VERIFICATION_PROMPT])