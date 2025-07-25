import cv2
import numpy as np
import os
from tqdm import tqdm

def clean_background_robust(img):
    # Step 1: Brightness threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

    # Step 2: HSV color filter for reddish/pink/purple
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_color = np.array([110, 20, 40])
    upper_color = np.array([170, 255, 255])
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

    # Step 3: Combine masks
    combined = cv2.bitwise_or(bright_mask, color_mask)

    # Step 4: Morph cleanup
    kernel = np.ones((10, 10), np.uint8)
    cleaned_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    # Step 5: Keep only largest contour
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    else:
        mask = cleaned_mask

    # Step 6: Apply mask to image
    white_bg = np.full_like(img, 255)
    mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    final_img = np.where(mask_3ch == 255, img, white_bg)

    return final_img, mask

def process_folder(input_folder, output_clean_folder, output_mask_folder):
    os.makedirs(output_clean_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png"))]
    for fname in tqdm(files, desc=f"Processing {input_folder}"):
        in_path = os.path.join(input_folder, fname)
        out_clean = os.path.join(output_clean_folder, fname)
        out_mask = os.path.join(output_mask_folder, fname.replace(".jpg", ".png"))

        img = cv2.imread(in_path)
        if img is None:
            print(f"❌ Skipped unreadable: {fname}")
            continue

        cleaned, mask = clean_background_robust(img)
        cv2.imwrite(out_clean, cleaned)
        cv2.imwrite(out_mask, mask)

if __name__ == "__main__":
    process_folder("data/OCA", "data/OCA_clean", "data/OCA_mask")
    process_folder("data/augmented_OCA", "data/augmented_OCA_clean", "data/augmented_OCA_mask")
