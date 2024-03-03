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
