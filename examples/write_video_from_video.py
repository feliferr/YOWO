import cv2
import numpy as np

# Create a VideoCapture object
video_path = "/Users/felipe.ferreira/Downloads/yowo-exp_datasets_AVA_videos_15min_9Y_l9NsnYE0.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

name, extension = video_path.split("/")[-1].split(".")
 
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = int(cv2.CAP_PROP_FRAME_HEIGHT)
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter(f"{name}_processed.{extension}", cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
while(True):
  ret, frame = cap.read()
  if ret == True:
    # Write the frame into the file
    out.write(frame)
  else:
    break 

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()