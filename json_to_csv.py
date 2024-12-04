import os
import json
import pandas as pd

def json_to_csv(folder_path: str, output_csv: str = "scene_analysis.csv"):
    """
    Convert all JSON files in a folder to a single CSV file with sorted scene names
    
    Args:
        folder_path: Path to folder containing JSON files
        output_csv: Name of output CSV file
    """
    # List to store all objects
    all_objects = []
    
    # Process each JSON file in the folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            # Read JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract scene name from filename
            pic_name = os.path.splitext(filename)[0]
            
            # Extract scene-level information
            scene_type = data['scene_type']
            
            # Process each object in the JSON
            for obj in data['objects']:
                object_data = {
                    'pic_name': pic_name,
                    'scene_type': scene_type,
                    'object_name': obj['object_name'],
                    'relationship_horizontal': obj['relationship_with_user_horizontal'],
                    'relationship_vertical': obj['relationship_with_user_vertical'],
                    'relationship_distance': obj['relationship_with_user_distance'],
                    'color': obj['color'],
                    'orientation': obj['orientation'],
                    'location': obj['location'],
                    'brightness': obj['brightness'],
                    'shape': obj['shape'],
                    'texture': obj['texture'],
                    'material': obj['material'],
                }
                all_objects.append(object_data)
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(all_objects)
    df = df.sort_values(['pic_name', 'object_name'])
    
    # Save to CSV
    df.to_csv(folder_path + "/" + output_csv, index=False)
    print(f"CSV file created successfully: {output_csv}")

json_to_csv("chatgpt-4o-latest")
json_to_csv("gpt-4o-2024-08-06")