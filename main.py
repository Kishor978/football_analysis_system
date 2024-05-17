from utils.video_utils import video_read,video_write
from trackers.tracker import Tracker
import cv2

def main():
    video_frames=video_read("data/video1.mp4")
    
    #inirialize tracker
    
    tracker=Tracker('models/best.pt')
    tracks=tracker.get_object_tracks(video_frames,
                                     read_from_stub=True,
                                     stub_path='stubs/trackers_stubs.pkl')
    
    # # save cropped image of a player
    # for track_id,player in tracks["players"][0].items():
    #     bbox=player['bbox']
    #     frame=video_frames[0]
    #     #crop player from frame
    #     cropped_player=frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
    #     #save cropped player
    #     cv2.imwrite(f"output/cropped_image.jpg",cropped_player)
    #     break
    
    #draw tracks on video
    output_video=tracker.draw_annotations(video_frames,tracks)
    
    video_write(output_video,"output/video1.avi")
    
if __name__ == "__main__":
    main()