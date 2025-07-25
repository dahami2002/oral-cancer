from PIL import Image
import os

def resize_images(src, dst, size=(64, 64)):
    os.makedirs(dst, exist_ok=True)
    for fname in os.listdir(src):
        if fname.lower().endswith((".jpg", ".png")):
            img = Image.open(os.path.join(src, fname)).convert("RGB")
            img = img.resize(size)
            img.save(os.path.join(dst, fname))

# Lesion: Resize OCA + OPMD
resize_images("data/gan_lesion_input", "data/gan_lesion_64/lesion")

# Non-lesion: Resize Benign + Healthy
resize_images("data/gan_nonlesion_input", "data/gan_nonlesion_64/nonlesion")

print("✅ Resized all images to 64×64.")
