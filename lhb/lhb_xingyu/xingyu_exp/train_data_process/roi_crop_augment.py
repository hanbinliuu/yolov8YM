import cv2
import numpy as np
import os
import time


def enhance_image(original_image_path, output_folder, num_enhancements=40, x=0, y=0, w=0, h=0):
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
        imCrop = enhanced_image[y: y + h, x: x + w]

        # Generate the output file path
        timestamp = int(time.time())  # Get current timestamp as an integer
        output_path = os.path.join(output_folder, f"enhanced_{timestamp}_{i + 1}.jpg")

        # Save the enhanced image
        cv2.imwrite(output_path, imCrop)


if __name__ == "__main__":
    # Input image path and output folder path
    input_image_path = r"C:\Users\Administrator\Desktop\test\Image_34.png"
    output_enhance_folder = r"C:\Users\Administrator\Desktop\img_enhancements"

    # Call the method to generate multiple enhanced images
    enhance_image(input_image_path, output_enhance_folder, num_enhancements=40,
                  x=775, y=1341, w=141, h=211
                  )
