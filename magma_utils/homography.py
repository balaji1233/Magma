import cv2
import numpy as np
import torch
import torch.nn.functional as F

def homography_warp(image, H, output_size):
    """
    Warps an image using a homography matrix H.

    Args:
        image (torch.Tensor): The image to be warped. Shape (C, H, W).
        H (torch.Tensor): The homography matrix. Shape (3, 3).
        output_size (tuple): The size of the output image (H, W).

    Returns:
        torch.Tensor: The warped image. Shape (C, H, W).
    """
    # Create a meshgrid of (x, y) coordinates
    h, w = output_size
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
    grid = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).float()  # Shape (H, W, 3)

    # Reshape grid to (H*W, 3) and apply homography matrix H
    grid = grid.view(-1, 3)  # Shape (H*W, 3)
    grid_warped = grid @ H.T  # Apply homography, Shape (H*W, 3)
    grid_warped = grid_warped[:, :2] / grid_warped[:, 2:]  # Normalize, Shape (H*W, 2)

    # Reshape grid_warped back to (H, W, 2)
    grid_warped = grid_warped.view(h, w, 2)

    # Normalize grid to the range [-1, 1]
    grid_warped[..., 0] = 2.0 * grid_warped[..., 0] / (w - 1) - 1.0
    grid_warped[..., 1] = 2.0 * grid_warped[..., 1] / (h - 1) - 1.0

    # Perform grid sampling
    grid_warped = grid_warped.unsqueeze(0)  # Add batch dimension, Shape (1, H, W, 2)
    image = image.unsqueeze(0)  # Add batch dimension, Shape (1, C, H, W)
    warped_image = F.grid_sample(image, grid_warped, mode='bilinear', align_corners=True)

    return warped_image.squeeze(0)  # Remove batch dimension

def calculate_homography(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create(5000)
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Calculate Homography matrix using RANSAC
    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    return H, matches, keypoints1, keypoints2

# Load your images
img1 = cv2.imread('/home/jianwyan/projects/ProjectWillow/azureblobs/vlpdatasets/epic-kitchen-magma/P01/P01_01_10/images/image_00000.png')
img2 = cv2.imread('/home/jianwyan/projects/ProjectWillow/azureblobs/vlpdatasets/epic-kitchen-magma/P01/P01_01_10/images/image_00010.png')

# Calculate homography matrix
H, matches, keypoints1, keypoints2 = calculate_homography(img1, img2)

# Print the homography matrix
print("Homography matrix:")
print(H)

# Draw matches (optional)
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Save image_matches to local
cv2.imwrite('matches.png', img_matches)

# Convert images to PyTorch tensors
image1_tensor = torch.from_numpy(img1.transpose(2, 0, 1)).float() / 255.0
image2_tensor = torch.from_numpy(img2.transpose(2, 0, 1)).float() / 255.0

# Compute the inverse of the homography matrix
H_inv = np.linalg.inv(H)

# Convert H_inv to a PyTorch tensor
H_inv_tensor = torch.tensor(H_inv, dtype=torch.float32)

# Define the output size (height, width) - typically the size of image2
output_size = (image2_tensor.shape[1], image2_tensor.shape[2])

# Perform inverse homography warping
warped_back_image = homography_warp(image1_tensor, H_inv_tensor, output_size)

# Convert the result to a numpy array for visualization (optional)
warped_back_image_np = warped_back_image.permute(1, 2, 0).cpu().numpy()

# save the warped back image to local
cv2.imwrite('warped_back_image.png', warped_back_image_np * 255.0)

# Get the dimensions of the images
height1, width1 = img1.shape[:2]
height2, width2 = img2.shape[:2]

# Warp the second image to align with the first
warped_img2 = cv2.warpPerspective(img2, H, (width1 + width2, height1))

# Place the first image in the stitched image
stitched_image = np.zeros_like(warped_img2)
stitched_image[0:height1, 0:width1] = img1

# Blend the images together
# stitched_image = cv2.addWeighted(stitched_image, 0.5, warped_img2, 0.5, 0)

# Save the stitched image to local
cv2.imwrite('stitched_image.png', stitched_image)
