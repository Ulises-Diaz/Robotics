from utils import read_video,save_video
from tracker import Tracker
import cv2
from asigner import TeamAssigner
import numpy as np
from player_ball_asigner import PlayerBallAssigner

def main():
    # Read video
    video_frames = read_video('input_videos/08fd33_4.mp4')
    


    # Initialize tracker 

    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub= True, stub_path= 'stubs/track_stubs.pkl')


    # interpolate ball positions

    tracks["ball"] =tracker.interpolate_ball(tracks["ball"])



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

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
