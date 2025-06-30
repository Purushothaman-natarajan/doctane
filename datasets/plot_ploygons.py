import random
import cv2
import json
import numpy as np
from pathlib import Path

def draw_polygons_on_images(
    detection_labels_path, images_dir, output_vis_dir, num_samples=10
):
    images_dir = Path(images_dir)
    output_vis_dir = Path(output_vis_dir)
    output_vis_dir.mkdir(parents=True, exist_ok=True)

    # Load detection labels
    with open(detection_labels_path, 'r', encoding='utf-8') as f:
        detection_labels = json.load(f)

    # Select random samples
    sample_keys = random.sample(list(detection_labels.keys()), min(num_samples, len(detection_labels)))

    for img_name in sample_keys:
        img_path = images_dir / img_name
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Unable to read image {img_name}")
            continue

        # Draw polygons
        for polygon in detection_labels[img_name]['polygons']:
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Save output
        save_path = output_vis_dir / f"vis_{img_name}"
        cv2.imwrite(str(save_path), img)
        print(f"✅ Saved visualized image: {save_path}")

# Call the visualization function
draw_polygons_on_images(
    detection_labels_path="./dataset/detection_labels.json",
    images_dir="./dataset/images",
    output_vis_dir="./visualized_samples",
    num_samples=30
)
