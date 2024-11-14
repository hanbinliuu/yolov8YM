import cv2
import numpy as np
import os
import time


def enhance_image(image_folder, output_folder, num_enhancements=20):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        # Get a list of all files in the image folder
    image_files = os.listdir(image_folder)

    for image_file in image_files:
        # Check if the file is an image file
        if image_file.endswith(".jpg") or image_file.endswith(".png") or image_file.endswith(".jpeg"):
            # Read the image
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)

            # Perform enhancements
            for i in range(num_enhancements):
                # Randomly generate brightness and contrast factors
                brightness_factor = np.random.uniform(0.5, 2)
                contrast_factor = np.random.uniform(0.5, 2)

                # Adjust image brightness and contrast
                enhanced_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_factor)

                # Generate the output file path
                timestamp = int(time.time())  # Get current timestamp as an integer
                output_path = os.path.join(output_folder, f"enhanced_{timestamp}{image_file}_{i + 1}.jpg")

                # Save the enhanced image
                cv2.imwrite(output_path, enhanced_image)


if __name__ == "__main__":

    input_image_folder = r"C:\Users\lhb\Desktop\croptest"
    output_enhance_folder = r"C:\Users\lhb\Desktop\lightenhance"
    enhance_image(input_image_folder, output_enhance_folder)
    temp = 1