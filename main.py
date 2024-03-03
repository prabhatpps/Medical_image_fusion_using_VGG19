import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg19


class VGG19(torch.nn.Module):
    def __init__(self, device='cpu'):
        """
        Initializes the VGG19 model for feature extraction.

        Parameters:
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        super(VGG19, self).__init__()
        # Load VGG19 features without pre-trained weights
        features = list(vgg19(weights=None).features)
        # Move features to the specified device
        if device == "cuda":
            self.features = nn.ModuleList(features).cuda().eval()
        else:
            self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        """
        Forward pass through the VGG19 model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            list: List of feature maps at a specific layer.
        """
        feature_maps = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            # Collect feature maps at the 4th layer
            if idx == 3:
                feature_maps.append(x)
        return feature_maps


class Fusion:
    def __init__(self, input):
        """
        Initializes the Fusion object.

        Parameters:
            input (list): List of input images.
        """
        # Initialize instance variables
        self.normalized_images = None
        self.input_images = input
        # Determine the device for computation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the VGG19 model
        self.model = VGG19(self.device)

    def fuse(self):
        """
        Performs image fusion.

        Returns:
            np.ndarray: Fused image.
        """
        # Convert input images to YCbCr format and transfer to tensors
        self.normalized_images = [-1 for img in self.input_images]
        self.YCbCr_images = [-1 for img in self.input_images]
        for idx, img in enumerate(self.input_images):
            if not self.gray_or_not(img):
                self.YCbCr_images[idx] = self.convert_RGB_to_YCbCr(img)
                self.normalized_images[idx] = self.YCbCr_images[idx][:, :, 0]
            else:
                self.normalized_images[idx] = img / 255.
        # Transfer images to PyTorch tensors
        self.transfer_image_to_tensor()
        # Perform fusion algorithm
        fused_img = self.fuse_algorithm()[:, :, 0]
        # Reconstruct fused image given RGB input images
        for idx, img in enumerate(self.input_images):
            if not self.gray_or_not(img):
                self.YCbCr_images[idx][:, :, 0] = fused_img
                fused_img = self.convert_YCbCr_to_RGB(self.YCbCr_images[idx])
                fused_img = np.clip(fused_img, 0, 1)
        return (fused_img * 255).astype(np.uint8)

    def fuse_algorithm(self):
        """
        Performs fusion algorithm.

        Returns:
            np.ndarray: Fused image.
        """
        with torch.no_grad():
            imgs_sum_maps = [-1 for tensor_img in self.images_to_tensors]
            # Compute sum maps for each input image
            for idx, tensor_img in enumerate(self.images_to_tensors):
                imgs_sum_maps[idx] = []
                feature_maps = self.model(tensor_img)
                for feature_map in feature_maps:
                    sum_map = torch.sum(feature_map, dim=1, keepdim=True)
                    imgs_sum_maps[idx].append(sum_map)
            max_fusion = None
            # Perform weighted fusion
            for sum_maps in zip(*imgs_sum_maps):
                features = torch.cat(sum_maps, dim=1)
                weights = self.softmax(F.interpolate(features,
                                                     size=self.images_to_tensors[0].shape[2:]))
                weights = F.interpolate(weights,
                                        size=self.images_to_tensors[0].shape[2:])
                current_fusion = torch.zeros(self.images_to_tensors[0].shape)
                for idx, tensor_img in enumerate(self.images_to_tensors):
                    current_fusion += tensor_img * weights[:, idx]
                if max_fusion is None:
                    max_fusion = current_fusion
                else:
                    max_fusion = torch.max(max_fusion, current_fusion)
            output = np.squeeze(max_fusion.cpu().numpy())
            if output.ndim == 3:
                output = np.transpose(output, (1, 2, 0))
            return output

    def convert_RGB_to_YCbCr(self, img_RGB):
        """
        Converts an RGB image to YCrCb format.

        Parameters:
            img_RGB (np.ndarray): RGB image.

        Returns:
            np.ndarray: YCbCr image.
        """
        img_RGB = img_RGB.astype(np.float32) / 255.
        return cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)

    def convert_YCbCr_to_RGB(self, img_YCbCr):
        """
        Converts a YCrCb image to RGB format.

        Parameters:
            img_YCbCr (np.ndarray): YCbCr image.

        Returns:
            np.ndarray: RGB image.
        """
        img_YCbCr = img_YCbCr.astype(np.float32)
        return cv2.cvtColor(img_YCbCr, cv2.COLOR_YCrCb2RGB)

    def gray_or_not(self, img):
        """
        Checks if an image is grayscale.

        Parameters:
            img (np.ndarray): Input image.

        Returns:
            bool: True if the image is grayscale, False otherwise.
        """
        if len(img.shape) < 3:
            return True
        if img.shape[2] == 1:
            return True
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if (b == g).all() and (b == r).all():
            return True
        return False

    def softmax(self, tensor):
        """
        Computes softmax output of a given tensor.

        Parameters:
            tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Softmax output.
        """
        tensor = torch.exp(tensor)
        tensor = tensor / tensor.sum(dim=1, keepdim=True)
        return tensor

    def transfer_image_to_tensor(self):
        """
        Transfers all input images to PyTorch tensors.
        """
        self.images_to_tensors = []
        for image in self.normalized_images:
            np_input = image.astype(np.float32)
            if np_input.ndim == 2:
                np_input = np.repeat(np_input[None, None], 3, axis=1)
            else:
                np_input = np.transpose(np_input, (2, 0, 1))[None]
            if self.device == "cuda":
                self.images_to_tensors.append(torch.from_numpy(np_input).cuda())
            else:
                self.images_to_tensors.append(torch.from_numpy(np_input))
