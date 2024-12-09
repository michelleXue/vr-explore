import os
from openai import OpenAI
from object_detection import detect_objects, DETECTION_PROMPT, VERIFICATION_PROMPT, SIMPLIFIED_SPATIAL_RELATIONSHIP_DETECTION_PROMPT
from same_object_detection import compare_objects, COMPARISON_PROMPT, VERIFICATION_PROMPT as COMPARISON_VERIFICATION_PROMPT
from object_coordinate_detection import detect_objects_coordinate
from itertools import combinations
import concurrent.futures

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

def process_images_in_folder(folder_path, detection_function, **kwargs):
    """
    Generic function to process images in a folder using specified detection function
    
    Args:
        folder_path: Path to the folder containing images
        detection_function: Function to process each image (detect_objects or detect_objects_coordinate)
        **kwargs: Additional arguments to pass to the detection function
    """
    # Create a list of tasks
    tasks = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('_1.png', '_6.png', '_11.png')):
            file_path = os.path.join(folder_path, filename)
            tasks.append(file_path)

    # Process tasks using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        for file_path in tasks:
            future = executor.submit(
                detection_function,
                file_path,
                call_api,
                **kwargs
            )
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(f"{detection_function.__name__} completed")
            except Exception as e:
                print(f"{detection_function.__name__} failed: {e}")

def same_object_detection_in_folder():
    # Define base features
    features = ["color", "shape", "orientation and location"]
    
    # Generate all possible combinations (including empty and single features)
    feature_combinations = [None]  # Start with None
    for r in range(1, len(features) + 1):
        for combo in combinations(features, r):
            feature_combinations.append(", ".join(combo))

    scenes = [
        # Easy scenes
        ("dataset-box/scene_easy_1_1.png", ["dataset-box/scene_easy_1_2.png", "dataset-box/scene_easy_1_4.png"]),
        ("dataset-box/scene_easy_2_1.png", ["dataset-box/scene_easy_2_2.png", "dataset-box/scene_easy_2_4.png"]),
        ("dataset-box/scene_easy_3_1.png", ["dataset-box/scene_easy_3_2.png", "dataset-box/scene_easy_3_4.png"]),
        # Medium scenes
        ("dataset-box/scene_medium_1_1.png", ["dataset-box/scene_medium_1_2.png", "dataset-box/scene_medium_1_4.png"]),
        ("dataset-box/scene_medium_2_1.png", ["dataset-box/scene_medium_2_2.png", "dataset-box/scene_medium_2_4.png"]),
        ("dataset-box/scene_medium_3_1.png", ["dataset-box/scene_medium_3_2.png", "dataset-box/scene_medium_3_4.png"]),
        # Hard scenes
        ("dataset-box/scene_hard_1_1.png", ["dataset-box/scene_hard_1_2.png", "dataset-box/scene_hard_1_4.png"]),
        ("dataset-box/scene_hard_2_1.png", ["dataset-box/scene_hard_2_2.png", "dataset-box/scene_hard_2_4.png"]),
        ("dataset-box/scene_hard_3_1.png", ["dataset-box/scene_hard_3_2.png", "dataset-box/scene_hard_3_4.png"])
    ]

    # Test each combination for each scene and comparison
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        for base_img, comparison_imgs in scenes:
            for comparison_img in comparison_imgs:
                for features in feature_combinations:
                    for box2_color in ["blue", "green"]:
                        future = executor.submit(
                            compare_objects,
                            base_img, 
                            comparison_img, 
                            call_api, 
                            box1_color="red", 
                            box2_color=box2_color, 
                            feature=features, 
                            prompts=[COMPARISON_PROMPT, COMPARISON_VERIFICATION_PROMPT]
                        )
                        futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print("mission completed")
            except Exception as e:
                print(f"mission failed: {e}")

if __name__ == "__main__":
    # same_object_detection_in_folder()
    
    # process_images_in_folder(
    #     "dataset", 
    #     detect_objects, 
    #     prompts=[SIMPLIFIED_SPATIAL_RELATIONSHIP_DETECTION_PROMPT, VERIFICATION_PROMPT]
    # )
    
    process_images_in_folder("dataset-compressed", detect_objects_coordinate)