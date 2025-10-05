from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
sys.path.append('../')
from utils import get_center_box, width_box
import cv2 


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections
    
    def get_object_track(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)
        
        tracks = {
            "team1_players": [],
            "team2_players": [],
            "ball": [],
        }
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            

            # this is necessary bcs classes are gives as numbers and we need strings
            cls_names_inv = {v: k for k, v in cls_names.items()}
            
            detection_supervision = sv.Detections.from_ultralytics(detection)
            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["team1_players"].append({})
            tracks["team2_players"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detections_with_tracks:
                bounding_box = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if class_id == cls_names_inv["team1_player"]:
                    tracks["team1_players"][frame_num][track_id] = {"bounding_box": bounding_box}
                
                if class_id == cls_names_inv["team2_player"]:
                    tracks["team2_players"][frame_num][track_id] = {"bounding_box": bounding_box}
            
            for frame_detection in detection_supervision:
                bounding_box = frame_detection[0].tolist()
                class_id = frame_detection[3]
                
                
                if class_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bounding_box": bounding_box}
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        
        return tracks

    def draw_ellipse(self, frame, bouding_box, color, track_id): 
        y2 = int(bouding_box[3])

        x_center , _= get_center_box(bouding_box)
        width = width_box(bouding_box)

        cv2.ellipse(frame, (x_center, y2), axes=(int(width), int(0.35*width)), angle=0.0, startAngle=-45, endAngle=235, color=color, thickness=2 , lineType=cv2.LINE_4) # Modifiy angels

        return frame
    
    

    def draw_annotations(self, video_frames, tracks): 
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames): 
            frame = frame.copy() # to not pollute video 

            player_team1_dict = tracks["team1_players"][frame_num] # dictionary of team players for each frame
            player_team2_dict = tracks["team2_players"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # draw players

            for track_id, player in player_team1_dict.items(): 
                frame = self.draw_ellipse(frame, player['bounding_box'], (0,0,255), track_id)

            output_video_frames.append(frame)

        return output_video_frames
