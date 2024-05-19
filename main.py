from utils.video_utils import video_read,video_write
from trackers.tracker import Tracker
from team_assignment import TeamAssigner,PlayerBallAssigner
from camera_movement import CameraMovementEstimator,PrespectiveTransformer
import cv2
import numpy as np

def main():
    video_frames=video_read("data/video1.mp4")
    
    #inirialize tracker
    
    tracker=Tracker('models/best.pt')
    tracks=tracker.get_object_tracks(video_frames,
                                     read_from_stub=True,
                                     stub_path='stubs/trackers_stubs.pkl')
    
    # get object positions
    tracker.add_positions_to_tracks(tracks)
    
    
    # camera movement estimation
    camera_movement_estimator=CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame=camera_movement_estimator.get_camera_movement(video_frames,read_from_stub=True,stub_path='stubs/camera_movement_stubs.pkl'),
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame[0])
    
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
            
    # ball aquisition
    player_assigner=PlayerBallAssigner()
    team_ball_control=[]
    for frame_num,player_track in enumerate(tracks['players']):
        ball_box=tracks['ball'][frame_num][1]['bbox']
        assinged_player=player_assigner.assign_ball_to_players(player_track,ball_box)
        
        if assinged_player !=-1:
            tracks['players'][frame_num][assinged_player]['has_ball']=True
            team_ball_control.append(tracks['players'][frame_num][assinged_player]['team'])
            
        else:
            team_ball_control.append(team_ball_control[-1])
    
    team_ball_control=np.array(team_ball_control)
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
    output_video=tracker.draw_annotations(video_frames,tracks,team_ball_control)
    
    # Drae the camera movement on the video
    output_video_frames=camera_movement_estimator.draw_camera_movement(output_video,camera_movement_per_frame)
    video_write(output_video_frames,"output/video1.avi")
    
if __name__ == "__main__":
    main()