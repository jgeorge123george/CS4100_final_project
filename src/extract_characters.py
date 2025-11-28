import cv2
import os

IMAGE_PATH = "/Users/josephgeorge/cs4100/CS4100_final_project/hello.png"
BOX_PATH = "/Users/josephgeorge/cs4100/CS4100_final_project/boxes.txt"
OUTPUT_DIR = "/Users/josephgeorge/cs4100/CS4100_final_project/testerac_output"

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the image
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("Image path is wrong: " + IMAGE_PATH)

h, w, _ = img.shape

boxes = []
with open(BOX_PATH, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()

    if len(parts) < 5:
        continue

    char = parts[0]
    x0, y0, x1, y1 = map(int, parts[1:5])

    y0_fixed = h - y0
    y1_fixed = h - y1

    y_top = min(y0_fixed, y1_fixed)
    y_bottom = max(y0_fixed, y1_fixed)

    crop = img[y_top:y_bottom, x0:x1]

    if crop.size == 0:
        continue

    boxes.append((char, crop))

for i, (char, crop_img) in enumerate(boxes):

    # Draw white bounding box inside the crop
    cv2.rectangle(
        crop_img,
        (0, 0),
        (crop_img.shape[1] - 1, crop_img.shape[0] - 1),
        (255, 255, 255),
        2
    )

    # Save to your desired output folder
    cv2.imwrite(f"{OUTPUT_DIR}/char_{i}.png", crop_img)

print(f"Saved {len(boxes)} raw cropped character images to {OUTPUT_DIR}.")

