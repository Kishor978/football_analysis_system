from utils.video_utils import video_read,video_write
from trackers.tracker import Tracker

def main():
    video_frames=video_read("data/video1.mp4")
    
    #inirialize tracker
    
    tracker=Tracker('models/best.pt')
    tracks=tracker.get_object_tracks(video_frames,
                                     read_from_stubs=True,
                                     stub_path='stubs/trackers_stubs.pkl')
    video_write(video_frames,"output/video1.avi")
    
if __name__ == "__main__":
    main()