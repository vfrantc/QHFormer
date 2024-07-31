import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import argparse

# Initialize the LPIPS model
loss_fn = lpips.LPIPS(net='alex')

# Function to load image
def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

# Function to compute metrics
def compute_metrics(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    psnr_value = psnr(img1_gray, img2_gray)
    ssim_value = ssim(img1_gray, img2_gray)
    
    img1_tensor = lpips.im2tensor(lpips.load_image_from_cv2(img1))
    img2_tensor = lpips.im2tensor(lpips.load_image_from_cv2(img2))
    lpips_value = loss_fn(img1_tensor, img2_tensor).item()

    return psnr_value, ssim_value, lpips_value

# Main function
def main(in_folder, gt_folder):
    image_extensions = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']
    gt_images = {os.path.splitext(f)[0]: os.path.join(gt_folder, f) for f in os.listdir(gt_folder) if os.path.splitext(f)[1] in image_extensions}
    
    psnr_total, ssim_total, lpips_total = 0.0, 0.0, 0.0
    count = 0

    for image_name in os.listdir(in_folder):
        name, ext = os.path.splitext(image_name)
        if name in gt_images:
            in_image_path = os.path.join(in_folder, image_name)
            gt_image_path = gt_images[name]
            
            in_image = load_image(in_image_path)
            gt_image = load_image(gt_image_path)
            
            if in_image is None or gt_image is None:
                print(f"Error loading images: {in_image_path} or {gt_image_path}")
                continue
            
            psnr_value, ssim_value, lpips_value = compute_metrics(in_image, gt_image)
            
            psnr_total += psnr_value
            ssim_total += ssim_value
            lpips_total += lpips_value
            count += 1
    
    if count > 0:
        psnr_avg = psnr_total / count
        ssim_avg = ssim_total / count
        lpips_avg = lpips_total / count
        
        print(f"Results for input folder: {in_folder} and ground truth folder: {gt_folder}")
        print(f"Average PSNR: {psnr_avg:.4f}")
        print(f"Average SSIM: {ssim_avg:.4f}")
        print(f"Average LPIPS: {lpips_avg:.4f}")
    else:
        print(f"No matching images found in input folder: {in_folder} and ground truth folder: {gt_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute average PSNR, SSIM and LPIPS for pairs of images with the same names.')
    parser.add_argument('--in_folder', type=str, required=True, help='Path to the input folder.')
    parser.add_argument('--gt_folder', type=str, required=True, help='Path to the ground truth folder.')
    
    args = parser.parse_args()
    
    main(args.in_folder, args.gt_folder)

