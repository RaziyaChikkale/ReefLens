import os

DATASET_PATH = "dataset"

print("\n========== DATASET CHECK ==========\n")

if not os.path.exists(DATASET_PATH):
    print("ERROR: dataset folder not found!")
    print("Create a folder named 'dataset' in the project folder.")
    exit()

total_images = 0

for root, dirs, files in os.walk(DATASET_PATH):

    image_files = [
        f for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if image_files:
        class_name = os.path.basename(root)

        print(f"{class_name}: {len(image_files)} images")

        total_images += len(image_files)

print("\nTotal images:", total_images)

if total_images == 0:
    print("No images found. Check your dataset folder.")
else:
    print("Dataset found successfully!")