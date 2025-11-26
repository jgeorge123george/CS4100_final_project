# CS4100_final_project

📦 Install Dependencies
Install Tesseract OCR

macOS

brew install tesseract


Windows
Download and install from:
https://github.com/UB-Mannheim/tesseract/wiki

Install Python Dependencies
pip install opencv-python


(NumPy installs automatically with OpenCV.)

Step 1 — Place Your Image

Place your handwriting image in the project directory 

Step 2 — Generate boxes.txt Using Tesseract

Run this command inside the project folder:

tesseract text.jpg stdout --psm 6 makebox > boxes.txt


If handwriting is messy or cursive, try:

tesseract text.jpg stdout --psm 7 makebox > boxes.txt


or:

tesseract text.jpg stdout --psm 13 makebox > boxes.txt


After running, you should see:

text.jpg
boxes.txt


boxes.txt contains bounding box coordinates for each detected character.

Step 3 — Extract 28×28 Character Images

Modify the IMAGE_PATH and BOX_PATH in extract_characters.py and run it. It will generate the individual png files for each
character: