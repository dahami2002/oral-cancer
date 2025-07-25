import os
import cv2
import albumentations as A
from tqdm import tqdm

# Paths
input_dir = "data/OCA"
output_dir = "data/augmented_OCA"
os.makedirs(output_dir, exist_ok=True)

# Desired final image count
target_count = 750
original_images = [f for f in os.listdir(input_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
original_count = len(original_images)
required_augmented = target_count - original_count

print(f"Original OCA images: {original_count}")
print(f"Need to generate: {required_augmented} new images")

# Define augmentation pipeline
augmentations = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=30, p=0.6),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.GaussianBlur(p=0.2),
    A.RandomGamma(p=0.3),
    A.HueSaturationValue(p=0.3)
])

# Calculate how many to generate per original image
aug_per_image = required_augmented // original_count + 1

# Augment and save
augmented_count = 0
pbar = tqdm(total=required_augmented)

for img_name in original_images:
    img_path = os.path.join(input_dir, img_name)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i in range(aug_per_image):
        if augmented_count >= required_augmented:
            break
        augmented = augmentations(image=image)['image']
        aug_name = f"{img_name.split('.')[0]}_aug{i+1}.jpg"
        save_path = os.path.join(output_dir, aug_name)
        cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
        augmented_count += 1
        pbar.update(1)

pbar.close()
print(f"✅ Done! {augmented_count} augmented OCA images created.")
