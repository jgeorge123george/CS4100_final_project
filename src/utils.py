import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"



def detect_spaces_from_boxes(box_lines):
    chars = []
    boxes = []

    for line in box_lines:
        parts = line.split()
        char = parts[0]
        x1, y1, x2, y2 = map(int, parts[1:5])
        chars.append(char)
        boxes.append((x1, x2))

    gaps = []
    for i in range(len(boxes) - 1):
        right = boxes[i][1]
        left_next = boxes[i+1][0]
        gaps.append(left_next - right)

    median_gap = sorted(gaps)[len(gaps)//2]
    threshold = 2 * median_gap

    result = chars[0]
    for i in range(1, len(chars)):
        gap = gaps[i-1]
        if gap > threshold:
            result += " "
        result += chars[i]

    return result
