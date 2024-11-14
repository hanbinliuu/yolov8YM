import cv2
import numpy as np
import os
from PIL import Image
import time


def enhance_image(original_image_path, output_folder, num_enhancements=40):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the original image
    base_image = cv2.imread(original_image_path)

    # Perform enhancements
    for i in range(num_enhancements):
        # Randomly generate brightness and contrast factors
        brightness_factor = np.random.uniform(0.5, 1)
        contrast_factor = np.random.uniform(0.5, 1)

        # Adjust image brightness and contrast
        enhanced_image = cv2.convertScaleAbs(base_image, alpha=contrast_factor, beta=brightness_factor)

        # Generate the output file path
        timestamp = int(time.time())  # Get current timestamp as an integer
        output_path = os.path.join(output_folder, f"enhanced_{timestamp}_{i + 1}.jpg")

        # Save the enhanced image
        cv2.imwrite(output_path, enhanced_image)


def crop_and_save(input_folder, output_folder, x, y, w, h):
    # Ensure the output folder exists; create if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all image files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Build the full paths for input and output files
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the original image
            original_image = Image.open(input_path)

            # Crop the image based on given coordinates and dimensions
            cropped_image = original_image.crop((x, y, x + w, y + h))

            # Save the cropped image to the new folder
            cropped_image.save(output_path)


if __name__ == "__main__":
    # Input image path and output folder path
    input_image_path =r"D:\xingyu_all_data\80\screw_rubber\Image_800.png"
    output_enhance_folder = r"C:\Users\Administrator\Desktop\img_enhancements"

    # Call the method to generate multiple enhanced images
    enhance_image(input_image_path, output_enhance_folder, num_enhancements=40)

    # Call the method to crop images from the enhancement output folder
    crop_and_save(r"C:\Users\Administrator\Desktop\img_enhancements", r"C:\Users\Administrator\Desktop\cropped_images",
                  x = 775, y = 1341, w = 141, h = 211)
