import cv2 


def read_video(video_path) : 
    cap = cv2.VideoCapture(video_path) # cap is a video reproducer. cap for capture 
    frames = [] # to save each frame of the video 
    while True: 
        ret , frame = cap.read() # ret is a boolean value that indicates wheter the frame was read or not
        if not ret: 
            break 
        frames.append(frame) # se agrega frame a la lista de frames
    return frames 

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24 , (output_video_frames[0].shape[1], output_video_frames[0].shape[0])) # output_video_frames[0].shape[0,1] is the height and width of first frame
     
    for frame in output_video_frames:  # to compress all frames of output_video_path in XVID format to save it
        out.write(frame)

    out.release()     