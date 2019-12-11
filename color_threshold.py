import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh_min=0, thresh_max=255):
    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(abs_sobel * 255 / np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    return binary_output


def hls_hsv_threshold(image, hls_thresh_min=0, hls_thresh_max=255, hsv_thresh_min=0, hsv_thresh_max=255):
    
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls_channel = hls[:, :, 2]
    hls_binary = np.zeros_like(hls_channel)
    hls_binary[(hls_channel >= hls_thresh_min) & (hls_channel <= hls_thresh_max)] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_channel = hsv[:, :, 2]
    hsv_binary = np.zeros_like(hsv_channel)
    hsv_binary[(hsv_channel >= hsv_thresh_min) & (hsv_channel <= hsv_thresh_max)] = 1

    binary_output = np.zeros_like(hls_channel)
    binary_output[(hls_binary == 1) & (hsv_binary == 1)] = 1
    
    return binary_output


def gaussian_blur(img, kernel=1):
    
    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(img, (kernel, kernel), 0)
    
    return blur


def combined_threshold(image):
    
    ksize = 5

    # Apply each of the thresh-holding functions
    #sobel
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh_min=50, thresh_max=255)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh_min=100, thresh_max=255)
    sobel_binary = cv2.bitwise_and(gradx, grady)
    #hls_hsv
    hls_hsv_binary = hls_hsv_threshold(image, hls_thresh_min=150, hls_thresh_max=255, hsv_thresh_min=200, hsv_thresh_max=255)
    #combined
    combined = cv2.bitwise_or(sobel_binary, hls_hsv_binary)
    #gaussian blur
    combined = gaussian_blur(combined, kernel=1)
    
    return combined


if __name__ == '__main__':

    images = glob.glob('./test_images/*.jpg')    
    
    # show result on test images
    for idx, filename in enumerate(images):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        threshold = combined_threshold(image)
    
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(image)
        ax1.set_title(os.path.basename(filename)+' Original', fontsize=12)
        ax2.imshow(threshold)
        ax2.set_title(os.path.basename(filename)+' Threshold', fontsize=12)
        plt.show()