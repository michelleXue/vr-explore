import os
import json
import pandas as pd

def object_detection_json_to_csv(folder_path: str, output_csv: str = "scene_analysis.csv"):
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
                    'relationship_with_user': obj['relationship_with_user_horizontal'] + " + " + obj['relationship_with_user_vertical'] + " + " + obj['relationship_with_user_distance'],
                    'color': obj['color'],
                    'orientation': obj['orientation'],
                    'location': obj['location'],
                    'brightness': obj['brightness'],
                    'shape': obj['shape'],
                    'texture': obj['texture'] + " + " + obj['material'],
                }
                all_objects.append(object_data)
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(all_objects)
    df = df.sort_values(['pic_name', 'object_name'])
    
    # Save to CSV
    df.to_csv(folder_path + "/" + output_csv, index=False)
    print(f"CSV file created successfully: {output_csv}")

def same_object_detection_json_to_csv(folder_path: str, output_csv: str = "comparison_analysis.csv"):
    """
    Convert all comparison JSON files in a folder to a single CSV file with sorted names
    
    Args:
        folder_path: Path to folder containing JSON files
        output_csv: Name of output CSV file
    """
    # List to store all comparisons
    all_comparisons = []
    
    # Process each JSON file in the folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.startswith('comparison') and filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            # Read JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract filenames from the comparison filename
            # Format: comparison_scene1_1_scene1_2_red_blue_shape.json
            parts = filename[11:-5].split('_')  # Remove 'comparison_' prefix and '.json' suffix
            image1_name = f"{parts[0]}_{parts[1]}"
            image2_name = f"{parts[2]}_{parts[3]}"
            box1_color = parts[4]
            box2_color = parts[5]
            feature = parts[6]

            if box1_color == "red":
                if box2_color == "blue":
                    is_same_object_ground_truth = False
                elif box2_color == "green":
                    is_same_object_ground_truth = True
            
            comparison_data = {
                'image1_name': image1_name,
                'image2_name': image2_name,
                'box1_color': box1_color,
                'box2_color': box2_color,
                'feature': feature,
                'is_same_object_ground_truth': is_same_object_ground_truth,
                'is_same_object': data['is_same_object'],
                'confidence_score': data['confidence_score'],
                'reasoning': data['reasoning'],
            }
            all_comparisons.append(comparison_data)
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(all_comparisons)
    df = df.sort_values(['feature', 'image1_name', 'image2_name'])
    
    # Calculate accuracy for each feature
    for feature in df['feature'].unique():
        feature_df = df[df['feature'] == feature]
        correct_predictions = (feature_df['is_same_object'] == feature_df['is_same_object_ground_truth']).sum()
        total_predictions = len(feature_df)
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Accuracy for {feature}: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
        
        # Calculate accuracy for same/different object cases
        for is_same in [True, False]:
            subset_df = feature_df[feature_df['is_same_object_ground_truth'] == is_same]
            correct = (subset_df['is_same_object'] == subset_df['is_same_object_ground_truth']).sum()
            total = len(subset_df)
            if total > 0:  # Avoid division by zero
                sub_accuracy = (correct / total) * 100
                same_diff = "same" if is_same else "different"
                print(f"  - For {same_diff} objects: {sub_accuracy:.2f}% ({correct}/{total})")
        print()  # Add blank line between features
    
    # Save to CSV
    df.to_csv(folder_path + "/" + output_csv, index=False)
    print(f"CSV file created successfully: {output_csv}")

if __name__ == "__main__":
    # object_detection_json_to_csv("exp4")
    same_object_detection_json_to_csv("same_object-reason")