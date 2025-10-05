

from utils import read_video, save_video
from trackers import Tracker

def main():
    # read video 
    video_frames = read_video('input_videos/nebraska-volleyball---you-have-to-see-this-insane-rally-between-nebraska--kentucky.mp4')

    
    # init tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_track(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    
    # Draw Output
    # Draw Obiject tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # save video 
    save_video(output_video_frames, 'output_videos/output.avi')

if __name__ == '__main__' : 
    main()