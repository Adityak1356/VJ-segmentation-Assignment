import os
import cv2
import numpy as np
from tqdm import tqdm
from itertools import combinations

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--iou_limit', nargs='?', const=0.1, default = 0.1, type=float)
args = parser.parse_args()
iou_limit = args.iou_limit
# Define colors for each of the 23 classes (0–22)
CLASS_COLORS = [
    (128, 0, 0),     # 0: back_bumper
    (139, 0, 139),   # 1: back_door
    (255, 0, 0),     # 2: back_glass
    (255, 69, 0),    # 3: back_left_door
    (255, 140, 0),   # 4: back_left_light
    (255, 255, 0),   # 5: back_light
    (85, 107, 47),   # 6: back_right_door
    (0, 255, 0),     # 7: back_right_light
    (0, 128, 0),     # 8: front_bumper
    (0, 255, 255),   # 9: front_door
    (0, 0, 255),     # 10: front_glass
    (0, 0, 128),     # 11: front_left_door
    (0, 100, 0),     # 12: front_left_light
    (186, 85, 211),  # 13: front_light
    (138, 43, 226),  # 14: front_right_door
    (255, 105, 180), # 15: front_right_light
    (205, 92, 92),   # 16: hood
    (112, 128, 144), # 17: left_mirror
    (0, 0, 0),       # 18: object
    (192, 192, 192), # 19: right_mirror
    (255, 20, 147),  # 20: tailgate
    (0, 191, 255),   # 21: trunk
    (255, 165, 0),   # 22: wheel
]

# Paths

dataset_path = 'dataset'  
sets = ['train', 'eval', 'test']

#Functions to handle overlapping segments

def polygon_to_mask(polygon, height, width):
    """Convert polygon (list of (x, y)) to binary mask."""
    mask = np.zeros((height, width, 3), dtype=np.uint8)    
    points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    points = points.astype(np.int32)
    cv2.fillPoly(mask, [points], color=(1,1,1))
    return np.array(mask[:,:,0], dtype=bool)

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0.0

def check_overlapping_segments(polygons, height, width, iou_threshold=iou_limit):
    """Returns True if any pair of segments has IoU >= threshold."""
    masks = [polygon_to_mask(poly, height, width) for poly in polygons]
    for i, j in combinations(range(len(masks)), 2):
        iou = calculate_iou(masks[i], masks[j])
        if iou >= iou_threshold:
            return True
    return False

#Data pre-processing

bad_overlapping_segmentation_masks_counter = 0
good_segmentation_masks_counter = 0
wrong_annotation_files_counter = 0

for set_name in sets:
    label_dir = os.path.join(dataset_path, set_name, 'labels')
    image_dir = os.path.join(dataset_path, set_name, 'images')
    mask_output_dir = os.path.join(dataset_path, set_name, 'masks')
    os.makedirs(mask_output_dir, exist_ok=True)

    for label_file in tqdm(os.listdir(label_dir), desc=f'Processing {set_name}'):
        if not label_file.endswith('.txt'):
            continue

        label_path = os.path.join(label_dir, label_file)
        image_name = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Image not found: {image_path}")
            continue

        h, w = image.shape[:2]
        mask = np.zeros((h, w, 3), dtype=np.uint8)

        with open(label_path, 'r') as f:
            polygons = []
            for line_num, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 6:
                    continue  # Not enough points

                try:
                    
                    class_id = int(float(parts[0]))
                    if class_id >= len(CLASS_COLORS):
                        print(f"Invalid class {class_id} in {label_file}, line {line_num+1}")
                        continue

                    coords = list(map(float, parts[1:]))

                    if len(coords) % 2 != 0:
                        print(f"Odd number of coordinates in {label_file}, line {line_num+1}")
                        continue

                    points = np.array(coords, dtype=np.float32).reshape(-1, 2)
                    points[:, 0] *= w
                    points[:, 1] *= h
                    points = points.astype(np.int32)
                    polygons.append(points)
                    
                    color = CLASS_COLORS[class_id]
                    cv2.fillPoly(mask, [points], color=color)

                except Exception as e:
                    print(f"Error processing {label_file}, line {line_num+1}: {e}")
                    continue
        
        output_path = os.path.join(mask_output_dir, label_file.replace('.txt', '.png'))
        cv2.imwrite(output_path, mask)        

        if(np.sum(mask)==0):
            os.remove(image_path)
            os.remove(label_path)
            os.remove(output_path)
            wrong_annotation_files_counter+=1
                
        elif(check_overlapping_segments(polygons, h, w, iou_threshold=0.1)):
            os.remove(image_path)
            os.remove(label_path)
            os.remove(output_path)
            bad_overlapping_segmentation_masks_counter+=1
        else:
            good_segmentation_masks_counter+=1
                
print("Data Processing Finished!")  
print("Summary:")
print("Total Masks Generated: "+str(bad_overlapping_segmentation_masks_counter+good_segmentation_masks_counter+wrong_annotation_files_counter))          
print("Masks removed due to incorrect annotations: "+str(wrong_annotation_files_counter))
print("Masks removed due to overlapping segments: "+ str(bad_overlapping_segmentation_masks_counter))
print("Remaining Masks: "+str(good_segmentation_masks_counter))

