import argparse
import os
import cv2
from fusion import Fusion
from evaluation import evaluate_metrics

def make_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image1_path', default=r'D:\testkk\New folder', help='Path to image1 images')
    parser.add_argument('--image2_path', default=r'D:\testkk\New folder (2)', help='Path to image2 images')
    parser.add_argument('-o', '--output_dir', default='./results2', help='Output directory for fused images')
    return parser.parse_args()

def fuse_images(images1, images2):
    # Initialize Fusion object with image1 and image2 images
    FU = Fusion([images1, images2])
    # Perform fusion
    fused_image = FU.fuse()
    return fused_image

if __name__ == '__main__':
    # Parse arguments
    args = make_args_parser()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of filenames for CT and PET images
    image1_files = os.listdir(args.image1_path)
    image2_files = os.listdir(args.image2_path)

    # Fuse images
    for image1_file, image2_file in zip(image1_files, image2_files):
        # Read image1 and image2 images
        image1 = cv2.imread(os.path.join(args.image1_path, image1_file), cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(os.path.join(args.image2_path, image2_file), cv2.IMREAD_GRAYSCALE)

        # Fuse the current pair of images
        fused_image = fuse_images(image1, image2)

        # Write fused image to output directory
        output_filename = os.path.splitext(image1_file)[0] + '_fused.png'
        cv2.imwrite(os.path.join(args.output_dir, output_filename), fused_image)

        # Evaluate metrics
        psnr_value, ssim_value, en_value, mi_value, mse_value, rmse_value = evaluate_metrics(image1, image2, fused_image)

        # Print metrics
        print(f"Metrics for {output_filename}:")
        print("PSNR:", psnr_value)
        print("SSIM:", ssim_value)
        print("Entropy:", en_value)
        print("Mutual Information:", mi_value)
        print("MSE:", mse_value)
        print("RMSE:", rmse_value)
        print("\n\n\n")
    print("Fussion is completed for all the images")
