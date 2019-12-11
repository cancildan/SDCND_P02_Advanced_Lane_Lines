import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

from camera_calibration import undistort_image
from perspective_transform import left_end, right_end
from perspective_transform import original2bird_eye
from color_threshold import combined_threshold


# Define conversions in x and y from pixels space to meters
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension


def calc_lane_lines(binary_warped):

    # Choose the number of sliding windows
    n_windows = 9
    window_height = np.int(binary_warped.shape[0] / n_windows)
    
    # Set the width of the windows +/- margin
    margin = 100
    
    # Set minimum number of pixels found to recenter window
    min_pixel = 50

    # Get starting point for the left and right lines
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = np.int((left_end + right_end) / 2)
    left_x_base = np.argmax(histogram[left_end:midpoint]) + left_end
    right_x_base = np.argmax(histogram[midpoint:right_end]) + midpoint
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    left_x_current = left_x_base
    right_x_current = right_x_base
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_indices = []
    right_lane_indices = []

    for window in range(n_windows):
        # Identify left and right window boundaries in x and y
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_left_x_low = left_x_current - margin
        win_left_x_high = left_x_current + margin
        win_right_x_low = right_x_current - margin
        win_right_x_high = right_x_current + margin
        cv2.rectangle(out_img, (win_left_x_low, win_y_low), (win_left_x_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_right_x_low, win_y_low), (win_right_x_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_indices = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                             (nonzero_x >= win_left_x_low) & (nonzero_x < win_left_x_high)).nonzero()[0]
        good_right_indices = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                              (nonzero_x >= win_right_x_low) & (nonzero_x < win_right_x_high)).nonzero()[0]
        left_lane_indices.append(good_left_indices)
        right_lane_indices.append(good_right_indices)

        # Recenter next window if necessary
        if len(good_left_indices) > min_pixel:
            left_x_current = np.int(np.mean(nonzero_x[good_left_indices]))
        if len(good_right_indices) > min_pixel:
            right_x_current = np.int(np.mean(nonzero_x[good_right_indices]))

    # Concatenate the arrays of indices
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    # Extract left and right line pixel positions
    left_x = nonzero_x[left_lane_indices]
    left_y = nonzero_y[left_lane_indices]
    right_x = nonzero_x[right_lane_indices]
    right_y = nonzero_y[right_lane_indices]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    # Generate x and y values for plotting
    plot_y = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

    left_points = np.dstack((left_x, left_y))
    right_points = np.dstack((right_x, right_y))
    curvature = get_curvature(left_x, left_y, np.max(plot_y))

    out_img[left_y, left_x] = [255, 0, 0]
    out_img[right_y, right_x] = [0, 0, 255]

    return {
        'left_points': left_points,
        'right_points': right_points,
        'left_fit_x': left_fit_x,
        'right_fit_x': right_fit_x,
        'left_fit': left_fit,
        'right_fit': right_fit,
        'plot_y': plot_y,
        'curvature': curvature,
        'out_img': out_img,
        'binary_warped': binary_warped
    }


def get_curvature(x, y, y_to_eval):
    
    # Fit new polynomials to x,y in world space
    left_fit = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
    y_to_eval *= ym_per_pix

    def cal_curvature(A, B, y):
        return ((1 + (2 * A * y + B) ** 2) ** 1.5) / np.absolute(2 * A)

    # Calculate the new radii of curvature
    left_curvature = cal_curvature(left_fit[0], left_fit[1], y_to_eval)

    return left_curvature


if __name__ == '__main__':

    images = glob.glob('./test_images/*.jpg')    
    
    # show result on test images
    for idx, filename in enumerate(images):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        undistorted = undistort_image(image)
        threshold = combined_threshold(undistorted)
        bird_eye = original2bird_eye(threshold)
        lane_line_params = calc_lane_lines(bird_eye)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(image)
        ax1.set_title(os.path.basename(filename)+' Original', fontsize=12)
        ax2.imshow(lane_line_params['out_img'])
        ax2.set_title(os.path.basename(filename)+' Calculated', fontsize=12)
        ax2.plot(lane_line_params['left_fit_x'], lane_line_params['plot_y'], color='yellow')
        ax2.plot(lane_line_params['right_fit_x'], lane_line_params['plot_y'], color='yellow')
        plt.show()