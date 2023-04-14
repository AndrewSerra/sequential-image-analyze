import cv2
import os
import argparse

def resize_images(input_directory, output_directory, size):
    for filename in os.listdir(input_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_filepath = os.path.join(input_directory, filename)
            output_filepath = os.path.join(output_directory, filename)
            img = cv2.imread(input_filepath)
            h, w, c = img.shape
            if h > w:
                diff = h - w
                left = diff // 2
                right = left + w
                img = img[left:right, :, :]
            elif w > h:
                diff = w - h
                top = diff // 2
                bottom = top + h
                img = img[:, top:bottom, :]
            img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_filepath, img_resized)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize images in a directory to the same size in square format without cropping')
    parser.add_argument('input_directory', type=str, help='Path to the directory containing input images')
    parser.add_argument('output_directory', type=str, help='Path to the directory where resized images will be saved')
    parser.add_argument('size', type=int, help='Size of the square image')
    args = parser.parse_args()

    resize_images(args.input_directory, args.output_directory, args.size)
