import os
import cv2
import numpy as np

# Define the number of steps for the gray scale conversion
NUM_GRAY_STEPS = 8

# Define the threshold value for the difference images
THRESHOLD_VALUE = 75

IMG_SET = "e2"
# Define the input and output directories
input_dir = f"inputs_{IMG_SET}"
output_dir = f"outputs_{IMG_SET}"
diff_dir = f"difference_imgs_{IMG_SET}"

# Create the output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(diff_dir, exist_ok=True)

# Get a list of all images that start with "trial_" and have a number at the end
input_files = [f for f in os.listdir(input_dir) if f.startswith("trial_") and f[-5:-4].isdigit()]

# Sort the files by their number
input_files = sorted(input_files, key=lambda f: int(f.split("_")[-1].split(".")[0]))

# Loop over the input files
prev_img = None
for i, input_file in enumerate(input_files):
    print(f"Working on image {input_file}")
    # Load the input image
    img = cv2.imread(os.path.join(input_dir, input_file))
    
    # Convert the image to gray scale with a certain number of steps
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a 3-channel grayscale image with the same intensity values as the original grayscale image
    gray_img_3ch = cv2.merge((gray_img, gray_img, gray_img))
    
    # Normalize the gray scale image
    gray_img_norm = cv2.normalize(gray_img, None, 0, NUM_GRAY_STEPS-1, cv2.NORM_MINMAX)
    
    # Scale the normalized gray scale image back to 0-255 range
    gray_img_norm = (gray_img_norm * (255 / (NUM_GRAY_STEPS-1))).astype('uint8')
    
    # Save the gray scale image
    output_file = os.path.join(output_dir, "gray_" + input_file)
    cv2.imwrite(output_file, gray_img_norm)
    
    # Create the difference image if this is not the first image
    if prev_img is not None:
        # Merge the blue channel of the previous image and the red channel of the current image
        blue_channel = cv2.split(prev_img)[0]
        red_channel = cv2.split(gray_img_3ch)[2]
        diff_img = cv2.merge((blue_channel, np.zeros_like(blue_channel), red_channel))
        
        # Convert the difference image to gray scale
        diff_gray_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
        
        # Apply a threshold to the difference image to highlight the change
        _, diff_gray_img = cv2.threshold(diff_gray_img, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        
        diff_output_file = os.path.join(diff_dir, "diff_" + input_file)
        cv2.imwrite(diff_output_file, diff_gray_img)
    
    # Set the previous image to the current image
    prev_img = gray_img_3ch
