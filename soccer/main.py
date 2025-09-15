from utils import read_video,save_video
from tracker import Tracker
import cv2
from asigner import TeamAssigner
import numpy as np
from player_ball_asigner import PlayerBallAssigner
from camera import CameraMovementEstimator
from perspective import Perspectivetransformer
from speed_distance_estimator import SpeedAndDistance_Estimator

def main():
    # Read video
    video_frames = read_video('input_videos/08fd33_4.mp4')
    
    # my computer died while processing, comment this for longer video output
    if len(video_frames) > 150:
        print(f"Large video detected ({len(video_frames)} frames), processing first 150 frames only")
        video_frames = video_frames[:150] 

    # Initialize tracker 

    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub= True, stub_path= 'stubs/track_stubs.pkl')
    print("Tracks obtained")

    # my computer does not support the num frames of the input video, eliminate this for longer video
    if len(tracks['players']) > len(video_frames):
            print(f"Reducing tracks from {len(tracks['players'])} to {len(video_frames)} frames")
            tracks['players'] = tracks['players'][:len(video_frames)]
            tracks['ball'] = tracks['ball'][:len(video_frames)]
            tracks['referees'] = tracks['referees'][:len(video_frames)]
    # camera movement estimator 
    
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_stub=True, stub_path='stubs/camera_movement_stub.pkl')

    def add_position_adjusted_to_tracks(tracks, camera_movement_per_frame):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bounding_box']
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    
                    if frame_num < len(camera_movement_per_frame):
                        camera_x, camera_y = camera_movement_per_frame[frame_num]
                        adjusted_x = center_x - camera_x
                        adjusted_y = center_y - camera_y
                    else:
                        adjusted_x, adjusted_y = center_x, center_y
                    
                    track_info['position_adjusted'] = [adjusted_x, adjusted_y]

    add_position_adjusted_to_tracks(tracks, camera_movement_per_frame)

    #view transformer
    view_transformer = Perspectivetransformer()
    view_transformer.add_transformed_position_2_tracks(tracks)


    # interpolate ball positions

    tracks["ball"] =tracker.interpolate_ball(tracks["ball"])

    # speed and distance estimator 
    speed_distance_estimator = SpeedAndDistance_Estimator()
    speed_distance_estimator.add_speed_distance_2_tracks(tracks)

    # save cropped image of player to color segmentation

    # for track_id, player in tracks["players"][0].items():
    #     bounding_box = player["bounding_box"]
    #     frame = video_frames[0]

    #     # crop bounding box from frame 
    #     cropped_image = frame[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]

    #     # saved cropped image
    #     cv2.imwrite(f'output_videos/cropped_img.jpg', cropped_image)
        

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                track["bounding_box"],
                                                player_id)
            
            tracks["players"][frame_num][player_id]['team'] = team
            tracks["players"][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    # Assign ball to player
    player_assigner = PlayerBallAssigner()
    team_ball_control= []


    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bounding_box = tracks["ball"][frame_num][1]["bounding_box"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bounding_box)

        if assigned_player != -1 : 
            tracks["players"][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]['team'])
        else : 
            team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)



    # Draw output
    # Draw object track
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # draw speed_distance
    speed_distance_estimator.draw_speed_distance(output_video_frames, tracks)


    # draw camera movement
    # output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
