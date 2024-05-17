from utils.video_utils import video_read,video_write
from trackers.tracker import Tracker
from team_assignment import TeamAssigner
import cv2

def main():
    video_frames=video_read("data/video1.mp4")
    
    #inirialize tracker
    
    tracker=Tracker('models/best.pt')
    tracks=tracker.get_object_tracks(video_frames,
                                     read_from_stub=True,
                                     stub_path='stubs/trackers_stubs.pkl')
    
    # interplatew ball positions
    tracks['ball']=tracker.interpolate_ball_positions(tracks['ball'])
    
    # Assgn Player teams
    team_assigner=TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],tracks['players'][0])
    
    for frame_num,player_track in enumerate(tracks['players']):
        for players_id,track in player_track.items():
            team=team_assigner.get_team(video_frames[frame_num],track['bbox'],players_id)
            tracks['players'][frame_num][players_id]['team']=team
            tracks['players'][frame_num][players_id]['team_color']=team_assigner.team_colors[team]
            
    
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