import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

IMAGE_SET = "e2"
# Define the path to the directory containing the images
img_dir = f"outputs_{IMAGE_SET}"
output_dir = f"heatmaps_{IMAGE_SET}"

# Create a new directory for the heatmaps
os.makedirs(output_dir, exist_ok=True)

# Get a list of all the image files in the directory
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

# Loop through each pair of back-to-back images
for i in range(len(img_files)-1):
   # Extract the frame numbers from the filenames
    frame_num1 = Path(img_files[i]).stem.split("_")[-1]
    frame_num2 = Path(img_files[i+1]).stem.split("_")[-1]

    # Load the images
    img1 = cv2.imread(os.path.join(img_dir, img_files[i]))
    img2 = cv2.imread(os.path.join(img_dir, img_files[i+1]))

    # Calculate the difference between the images
    diff = cv2.absdiff(img1, img2)

    # Convert the difference image to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Create a heatmap plot of the difference
    plt.imshow(gray, cmap='hot', interpolation='nearest')
    plt.title("Frame {0} vs Frame {1}".format(frame_num1, frame_num2))
    plt.savefig(os.path.join(output_dir, "heatmap_{0}_{1}.png".format(frame_num1, frame_num2)))
    plt.close()
