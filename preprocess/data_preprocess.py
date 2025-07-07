import os
import argparse
import pandas as pd
from PIL import Image
import numpy as np
import cv2

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image.load()
    img_np = np.array(image)

    gray = np.array(image.convert("L"))

    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    foreground_ratio = np.sum(clean_mask == 255) / clean_mask.size
    if foreground_ratio < 0.5:
        clean_mask = cv2.bitwise_not(clean_mask)

    if img_np.ndim == 2:
        masked_img = cv2.bitwise_and(img_np, clean_mask)
    else:
        mask_3ch = cv2.merge([clean_mask] * 3)
        masked_img = cv2.bitwise_and(img_np, mask_3ch)

    return masked_img

def batch_mask_images(input_folder, output_folder, output_csv):
    os.makedirs(output_folder, exist_ok=True)

    records = []

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"):
            parts = filename.split('_')
            if len(parts) == 2 and parts[0].startswith('slide') and parts[1].startswith('core'):
                try:
                    slide_number = int(parts[0][5:])
                    core_number = int(parts[1][4:7])
                except ValueError:
                    continue

                image_path = os.path.join(input_folder, filename)
                masked_image = process_image(image_path)

                new_filename = f"slide{slide_number:03d}_core{core_number:03d}_masked.png"
                mask_path = os.path.join(output_folder, new_filename)
                Image.fromarray(masked_image).save(mask_path)

                records.append({
                    "slide": slide_number,
                    "core": core_number,
                    "path": os.path.abspath(image_path),
                    "mask_path": os.path.abspath(mask_path)
                })

    # Save DataFrame
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"✅ Batch processing complete. Metadata saved to: {output_csv}")

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess slide_core images by masking background.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing input images")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save masked output images")
    parser.add_argument("--output_csv", type=str, default="processed_metadata.csv", help="CSV file to save metadata (default: processed_metadata.csv)")
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_args()
    batch_mask_images(opt.input_folder, opt.output_folder, opt.output_csv)
