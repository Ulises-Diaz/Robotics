from ultralytics import YOLO 
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()


    
    def detect_frames(self, frames):  # we recieve all detections of all frames in a video 
        batch_size = 20  # analyze 20 frames at a time. intead of analyzing all of them 
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1)  # analyzes from 0-20, then 20-40, [i:i+batch_size]
            detections += detections_batch  # += to have all detections inside one list. If using .append() we would have lists in a list, this will crash the tracking couse 
            # tracking only want all detections in a frame and not a list of detections, 
            break 
        return detections

    def get_object_track(self, frames) : 
        
        detections = self.detect_frames(frames) 

        for frame_num , detection in enumerate(detections) :  # looping over detections and indexing in a list with frame_num
            cls_names = detection.names 


            # convert detection to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #detection_with_tracks = self.tracker.update_with_detections(detection_supervision) 

            print(detection_supervision) 

