## Self-Driving Car Engineer Nanodegree Program

## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image1]: ./output_images/calibration1_undistorted.png "Undistorted"
[image2]: ./output_images/test1_undistorted.png "Road Transformed"
[image3]: ./output_images/test1_threshold.png "Binary Example"
[image4]: ./output_images/test1_bird_eye.png "Warp Example"
[image5]: ./output_images/test1_calculated.png "Fit Visual"
[image6]: ./output_images/r_curve.png "Curve"
[image7]: ./output_images/test1_result.png "Output"
[video1]: ./project_video_output.mp4 "Video"

## 

### Camera Calibration

##### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `camera_calibration.py` 

Every camera would has a little bit of distortion in the images at least, therefore calibration should have done to correct images to make decisions by using them. OpenCV provide calibration functions to undistort the images by using chessboard images taken by the this camera.

I started by using given chessboard images taken at different angles. Each image is grayscaled and sent into  `findChessboardCorners` . The resulting "object points" are the (x, y, z) coordinates of the chessboard corners. And the output is displayed and shown to the user. Finally the corner points are sent to `calibrateCamera` to get resulting image points and object points. This dictionary is then saved to use in undistortion for further steps.

As a final step use `undistort()` to see an undistorted chessboard, as shown below.

![alt text][image1]

### Pipeline (single images)

##### 1. Provide an example of a distortion-corrected image.

I used `calibration_params.p` from previous output to undistort the image using `undistort()`. The undistortion  is not obvious as applied chessboard images, but if you look close to the edges you can see the difference.

You can see original and undistorted image below:
![alt text][image2]

##### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is contained in `color_threshlod.py`

The second part of the pipeline is to find possible pixels of lane lines by tresholding with different filters. I applied below filters:

- Apply two Sobel filters in the x and y directions
  - Gradient on x-axis with sobel filter size of 5 and threshold of 50 and 255
  - Gradient on y-axis with sobel filter size of 5 and  threshold of 100 and 255
- Do a bitwise AND function for the two sobel filters
- Apply two color masks in HLS and HSV color space
  - The HLS mask uses a threshold of 150 and 255
  - The HSV mask uses a threshold of 200 and 255
- Do a bitwise AND function for the two color filters
- Finally apply a gaussian blur on the outputted image with a kernel of 1



![alt text][image3]

##### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for this step is contained in `perspective_transform.py`

The third part of the pipeline is to create a bird-eye view by using perspective transform, which allow us  to compute curvature in further steps. I used `getPerspectiveTransform` and `warpPerspective` OpenCV functions.

This resulted in the following source and destination points

|  Source   | Destination |
| :-------: | :---------: |
| 580, 460  |   320, 0    |
| 203, 720  |  320, 720   |
| 1127, 720 |  960, 720   |
| 700, 460  |   960, 0    |

Instead of determining source and destination points programmatically, I decided to selecting points manually by assuming that the camera position will remain constant and that the road in the videos will remain relatively flat.

You can see the results of the perspective transform below:

![alt text][image4]

##### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this step is contained in `find_lane_line.py`

The fourth part of pipeline is to find lane line pixels and fit it with a 2nd order polynomial. I used sliding window method  split image into multiple layers, locate window at the most intense area with histogram and mark those inside windows as lane line points.

![alt text][image5]

##### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step is contained in lines 115-127 of `find_lane_line.py`

The curvature of a 2nd order polynomial could be computed by:

![alt text][image6]

The distance from center has two assumptions about the input video which are can be seen below:

- The car position is assumed to be in the mid of the image
- The lane width follows US regulation (3.7m)

By scaling between pixel and meter, the final result of curvature and relative vehicle position are added to the top left corner.

##### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this step is contained in lines `image_pipeline.py`

Here is an example of the final result on a single image:

![alt text][image7]

---

### Pipeline (video)

##### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The code for this step is contained in lines `video_pipeline.py`

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

##### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

First of all I would say that it's really advance lane finding if I compare with the first project, it really forced me to clarify some OpenCV functions usage. While I was working on some tunning steps it was really hard to adjusting specific parameter. Especially changes in lighting, shadows and color conditions created some unsuccessful outputs. And also I would say, the solution is not optimal for all lane line finding problems while my pipeline is not capable to solve challenge videos. And one specific error I faced  like *TypeError(“expected non-empty vector for x”)* which was related with frames that don't have right lane in warped lane, so `find_lane_line()` couldn’t fit the right lane. I didn't pursue this error while just faced for challenge videos.

The most thing I want to improve later is the threshold, this would create more robust algorithm. Now it's not robust enough to shadow and different light conditions. For example to apply dynamic thresholding for different horizontal slices of the image would improve results. Another potential improvement would be using deep learning which would able to identify lane lines.

Overall, it was a very challenging project, but I now have better knowhow for computer vision then before.