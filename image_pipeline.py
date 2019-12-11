import numpy as np
import cv2
import glob
import os
import argparse
import matplotlib.pyplot as plt

from camera_calibration import undistort_image
from color_threshold import combined_threshold
from perspective_transform import original2bird_eye, invert_matrix
from find_lane_line import calc_lane_lines, xm_per_pix


def process_image(image):
    
    undistorted = undistort_image(image)
    threshold = combined_threshold(undistorted)
    bird_eye = original2bird_eye(threshold)
    lane_line_params = calc_lane_lines(bird_eye)
    result = draw_lane_lines(undistorted, lane_line_params)

    f, axarr = plt.subplots(2, 2, figsize=(30, 15))
    axarr[0, 0].imshow(undistorted)
    axarr[0, 0].set_title('Undistorted')
    axarr[0, 1].imshow(threshold, cmap='gray')
    axarr[0, 1].set_title('Threshold')
    axarr[1, 0].imshow(lane_line_params['out_img'])
    axarr[1, 0].plot(lane_line_params['left_fit_x'], lane_line_params['plot_y'], color='yellow')
    axarr[1, 0].plot(lane_line_params['right_fit_x'], lane_line_params['plot_y'], color='yellow')
    axarr[1, 0].set_title('Calculated')
    axarr[1, 1].imshow(result)
    axarr[1, 1].set_title('Result')
    plt.show()

    return result


def draw_lane_lines(undistorted, params):
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(params['binary_warped']).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Fill the part between lane lines with green
    # Recast the x and y points into usable format for cv2.fillPoly()
    fit_pts_left = np.array([np.transpose(np.vstack([params['left_fit_x'], params['plot_y']]))])
    fit_pts_right = np.array([np.flipud(np.transpose(np.vstack([params['right_fit_x'], params['plot_y']])))])
    fit_pts = np.hstack((fit_pts_left, fit_pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([fit_pts]), (0, 255, 0))
    
    # Inverse transform and combine the result with the original image
    new_warp = cv2.warpPerspective(color_warp, invert_matrix, undistorted.shape[1::-1])
    # Combine the result with the original image
    image = cv2.addWeighted(undistorted, 1, new_warp, 0.3, 0)

    # Add curvature text to the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, 'Curve radius  = {:.2f} m'.format(params['curvature']),
        (30, 40), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # Add vehicle position to the image
    left_fit_point = np.float64([[[params['left_fit_x'][-1], params['plot_y'][-1]]]])
    right_fit_point = np.float64([[[params['right_fit_x'][-1], params['plot_y'][-1]]]])
    left_fit_in_original = cv2.perspectiveTransform(left_fit_point, invert_matrix)
    right_fit_in_original = cv2.perspectiveTransform(right_fit_point, invert_matrix)

    lane_mid = .5 * (left_fit_in_original + right_fit_in_original)[0, 0, 0]
    vehicle_mid = image.shape[1] / 2
    dx = (vehicle_mid - lane_mid) * xm_per_pix * 100

    cv2.putText(image, 'Vehicle position = {:.2f} cm {} of center'.format(abs(dx),
        'left' if dx < 0 else 'right'),(30, 80), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

    return image


if __name__ == '__main__':
    
    images = glob.glob('./test_images/*.jpg')
    
    # show results on test images
    for idx, filename in enumerate(images):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = process_image(image)