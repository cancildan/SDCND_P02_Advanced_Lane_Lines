from moviepy.editor import VideoFileClip

from camera_calibration import undistort_image
from color_threshold import combined_threshold
from perspective_transform import original2bird_eye
from find_lane_line import calc_lane_lines
from image_pipeline import draw_lane_lines

def process_video_frame(image):
    
    global prev_fit

    undistorted = undistort_image(image)
    threshold = combined_threshold(undistorted)
    bird_eye = original2bird_eye(threshold)
    lane_line_params = calc_lane_lines(bird_eye)

    result = draw_lane_lines(undistorted, lane_line_params)
    
    return result


if __name__ == '__main__':

    clip1 = VideoFileClip('project_video.mp4')
    vid_clip = clip1.fl_image(process_video_frame)
    vid_clip.write_videofile('project_video_output.mp4', audio=False)
    
    '''clip2 = VideoFileClip('challenge_video.mp4')
    vid_clip = clip2.fl_image(process_video_frame)
    vid_clip.write_videofile('challenge_video_output.mp4', audio=False)
    
    clip3 = VideoFileClip('harder_challenge_video.mp4')
    vid_clip = clip3.fl_image(process_video_frame)
    vid_clip.write_videofile('harder_challenge_video_output.mp4', audio=False)'''