import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob
import os


def camera_calibration():

    # prepare object points
    nx = 9#TODO: enter the number of inside corners in x
    ny = 6#TODO: enter the number of inside corners in y
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d points in real world space
    img_points = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')
    print("Processing {} images".format(len(images)))
        
    # Make plotlayout for images  
    fig, axs = plt.subplots(5,4, figsize=(20, 9))
    fig.subplots_adjust(hspace = .5)
    axs = axs.ravel()

    # Step through the list and search for chessboard corners
    for idx, filename in enumerate(images):
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret:
            obj_points.append(objp)
            img_points.append(corners)
            
            # Draw and display the corners
            dist_img = cv2.drawChessboardCorners(image, (9,6), corners, ret)
            axs[idx].set_title(os.path.basename(filename), fontsize=10)
            axs[idx].axis('off')
            axs[idx].imshow(dist_img)
            cv2.imwrite("./output_images/chessboard{}.jpg".format(idx), image)

        else:
            print("Could not find chessboard corners for {}".format(os.path.basename(filename)))
            axs[idx].set_title(os.path.basename(filename), fontsize=10)
            axs[idx].imshow(image)

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image.shape[1::-1], None, None)
    
    return mtx, dist


def set_get_calibration_params():

    filename = 'calibration_params.p'
    if os.path.exists(filename):
        return pickle.load(open(filename, 'rb'))
    else:
        mtx, dist = camera_calibration()
        pickle.dump((mtx, dist), open(filename, 'wb'))
        
        return mtx, dist


def undistort_image(image):    

    mtx, dist = set_get_calibration_params()
    undistort_image = cv2.undistort(image, mtx, dist, None, mtx)
    
    return undistort_image


if __name__ == '__main__':

    images = glob.glob('./test_images/*.jpg')
   
    #undistort all calibration images
    for idx, filename in enumerate(images):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        undistort = undistort_image(image)
    
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(image)
        ax1.set_title(os.path.basename(filename)+' Original', fontsize=12)
        ax2.imshow(undistort)
        ax2.set_title(os.path.basename(filename)+' Undistorted', fontsize=12)
        plt.show()