import cv2

IMAGE_PATH = "/Users/josephgeorge/cs4100/CS4100_final_project/hello.png"
BOX_PATH = "/Users/josephgeorge/cs4100/CS4100_final_project/boxes.txt"

# Load the image
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("Image path is wrong: " + IMAGE_PATH)

h, w, _ = img.shape

# Read box lines
boxes = []
with open(BOX_PATH, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()

    # Expect: char x0 y0 x1 y1 page
    if len(parts) < 5:
        continue

    char = parts[0]
    x0, y0, x1, y1 = map(int, parts[1:5])

    # Tesseract uses bottom-left origin → flip the y-axis for OpenCV
    y0 = h - y0
    y1 = h - y1
    y_top = min(y0, y1)
    y_bottom = max(y0, y1)

    crop = img[y_top:y_bottom, x0:x1]

    if crop.size == 0:
        continue

    # Convert to grayscale and resize to 28×28 for the CNN
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    char28 = cv2.resize(gray, (28, 28))

    boxes.append((char, char28))

# Save cropped 28×28 character images
for i, (char, char_img) in enumerate(boxes):
    cv2.imwrite(f"char_{i}_{char}.png", char_img)

print(f"Saved {len(boxes)} character images.")