import cv2

video_file = "/home/jianwyan/projects/ProjectWillow/azureblobs/echelondata/datasets/epic_kitchens/release_2022/P27/videos/P27_105.MP4"

# read video
cap = cv2.VideoCapture(video_file)
# get frame count
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# get fps
fps = cap.get(cv2.CAP_PROP_FPS)
# get frame size
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# extract frame at 29071
cap.set(cv2.CAP_PROP_POS_FRAMES, 29071)
ret, frame = cap.read()
import pdb; pdb.set_trace()