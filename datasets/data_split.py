import shutil
import random
import json
from pathlib import Path
from tqdm import tqdm

def split_dataset(
    dataset_dir,
    output_base_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    random.seed(seed)

    dataset_dir = Path(dataset_dir)
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_dir / "images"
    word_crops_dir = dataset_dir / "filtered_recog_images"

    # Read labels
    with open(dataset_dir / "detection_labels.json", 'r', encoding='utf-8') as f:
        detection_labels = json.load(f)
    with open(dataset_dir / "filtered_recog_labels.json", 'r', encoding='utf-8') as f:
        recognition_labels = json.load(f)

    # List of image names
    all_images = list(detection_labels.keys())
    random.shuffle(all_images)

    train_count = int(train_ratio * len(all_images))
    val_count = int(val_ratio * len(all_images))
    test_count = len(all_images) - train_count - val_count

    train_images = all_images[:train_count]
    val_images = all_images[train_count:train_count + val_count]
    test_images = all_images[train_count + val_count:]

    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    print(f"Total images: {len(all_images)}")
    print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}\n")

    for split, split_images in splits.items():
        # Create folders
        split_dir = output_base_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        split_images_dir = split_dir / "images"
        split_word_crops_dir = split_dir / "word_crops"
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_word_crops_dir.mkdir(parents=True, exist_ok=True)

        split_detection_labels = {}
        split_recognition_labels = {}

        print(f"\nProcessing split: {split} ({len(split_images)} images)")
        for img_name in tqdm(split_images, desc=f"Copying {split} images"):
            # Copy full image
            src_img = images_dir / img_name
            dst_img = split_images_dir / img_name
            shutil.copy(src_img, dst_img)

            # Save detection label
            split_detection_labels[img_name] = detection_labels[img_name]

            # Find all associated word crops
            prefix = Path(img_name).stem + "_"
            word_crop_files = list(word_crops_dir.glob(f"{prefix}*.jpg"))
            for crop_file in word_crop_files:
                shutil.copy(crop_file, split_word_crops_dir / crop_file.name)
                if crop_file.name in recognition_labels:
                    split_recognition_labels[crop_file.name] = recognition_labels[crop_file.name]

        # Save split JSONs
        with open(split_dir / "detection_labels.json", 'w', encoding='utf-8') as f:
            json.dump(split_detection_labels, f, indent=2)

        with open(split_dir / "recognition_labels.json", 'w', encoding='utf-8') as f:
            json.dump(split_recognition_labels, f, indent=2)

        print(f"✅ {split}: {len(split_images)} full images copied")
        print(f"✅ {split}: {len(split_recognition_labels)} word crops copied")


# Split dataset ::
split_dataset(
    dataset_dir="/data/purushothaman.n/ocr/dataset_t1/",
    output_base_dir="/data/purushothaman.n/ocr/dataset_split_t1/",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
