import argparse
import cv2
from fusion import Fusion
from evaluation import evaluate_metrics


def make_args_parser():
    """
    Creates an argument parser for command-line arguments.

    Returns:
        ArgumentParser: Argument parser object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', nargs='+', type=str,
                        default=['DB/DB/CT/1.jpg',
                                 'DB/DB/PET/1.jpg'],
                        help='Define path to images')
    parser.add_argument('-o', '--output', default='./results2/output.jpg',
                        help='Define output path of fused image')
    return parser.parse_args()


def fuse_images(images1, images2):
    """
    Fuses two input images.

    Parameters:
        images1 (np.ndarray): First input image.
        images2 (np.ndarray): Second input image.

    Returns:
        np.ndarray: Fused image.
    """
    FU = Fusion([images1, images2])
    fused_image = FU.fuse()
    return fused_image


def print_args(args):
    """
    Prints the command-line arguments.

    Parameters:
        args (Namespace): Parsed command-line arguments.
    """
    print("Running with the following configuration")
    args_map = vars(args)
    for key in args_map:
        print('\t', key, '-->', args_map[key])
    print()


if __name__ == '__main__':
    args = make_args_parser()
    print_args(args)

    input_image1_loc = args.images[0]
    input_image2_loc = args.images[1]

    input_image1_rgb = cv2.imread(input_image1_loc)
    input_images2_rgb = cv2.imread(input_image2_loc)
    fused_image_rgb = fuse_images(input_image1_rgb, input_images2_rgb)

    cv2.imwrite(args.output, fused_image_rgb)

    input_image1_grayscale = cv2.imread(input_image1_loc, cv2.IMREAD_GRAYSCALE)
    input_images2_grayscale = cv2.imread(input_image2_loc, cv2.IMREAD_GRAYSCALE)
    fused_image_grayscale = fuse_images(input_image1_grayscale, input_images2_grayscale)

    psnr_value, ssim_value, en_value, mi_value, mse_value, rmse_value = evaluate_metrics(input_image1_grayscale,
                                                                                         input_images2_grayscale,
                                                                                         fused_image_grayscale)

    print("PSNR:", psnr_value)
    print("SSIM:", ssim_value)
    print("Entropy:", en_value)
    print("Mutual Information:", mi_value)
    print("MSE:", mse_value)
    print("RMSE:", rmse_value)
    print("\n\n")
    print("Fussion is completed for the given images")



## For Batch Processing
'''import argparse
import os
import cv2
from fusion import Fusion
from evaluation import evaluate_metrics


def make_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image1_path', default=r'DB/DB/CT', help='Path to image1 images')
    parser.add_argument('--image2_path', default=r'DB/DB/PET', help='Path to image2 images')
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

        # for rgb
        image1_rgb = cv2.imread(os.path.join(args.image1_path, image1_file))
        image2_rgb = cv2.imread(os.path.join(args.image2_path, image2_file))
        fused_image_rgb = fuse_images(image1_rgb, image2_rgb)

        # Write fused image to output directory
        output_filename = os.path.splitext(image1_file)[0] + '_fused.png'
        cv2.imwrite(os.path.join(args.output_dir, output_filename), fused_image_rgb)

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
'''