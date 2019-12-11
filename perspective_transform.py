import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import glob
import os

from camera_calibration import undistort_image
from color_threshold import combined_threshold

# source matrix
src = np.array([[580, 460],
                [203, 720],
                [1127, 720],
                [700, 460]], dtype=np.float32)

# target matrix to transform to
dst = np.array([[320, 0],
                [320, 720],
                [960, 720],
                [960, 0]], dtype=np.float32)

left_end, right_end = 250, 1080

# transform matrix
trans_matrix = cv2.getPerspectiveTransform(src, dst)
invert_matrix = cv2.getPerspectiveTransform(dst, src)

def original2bird_eye(image):        
    bird_eye = cv2.warpPerspective(image, trans_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)    
    return bird_eye


if __name__ == '__main__':
   
    images = glob.glob('./test_images/*.jpg')    
    
    # show result on test images
    for idx, filename in enumerate(images):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        undistorted = undistort_image(image)
        threshold = combined_threshold(undistorted)
        bird_eye = original2bird_eye(image)
    
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title(os.path.basename(filename)+' Original', fontsize=12)
        ax1.imshow(image, cmap='gray')
        for point in src:
            ax1.plot(*point, '.')
        ax2.set_title(os.path.basename(filename)+' Bird Eye', fontsize=12)
        ax2.imshow(bird_eye, cmap='gray')
        for point in dst:
            ax2.plot(*point, '.')