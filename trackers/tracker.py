from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
from utils import get_bbox_width,get_center

class Tracker:
    def __init__(self, model_path):
        self.model=YOLO(model_path)
        self.tracker=sv.ByteTrack()
        
    def detect_frames(self,frames):
        batch_size=15
        detections=[]
        for i in range(0,len(frames),batch_size):
            detections_batch=self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections=detections+detections_batch
        return detections
        
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks    
    
    def draw_ellipse(self,frame,bounding_box,color,track_id=None):
        y2=int(bounding_box[3])
        x_center,_=get_center(bounding_box)
        width=get_bbox_width(bounding_box)
        cv2.ellipse(frame,(x_center,y2),
                    axes=(int(width),int(0.35*width)),
                    angle=0,
                    startAngle=45,
                    endAngle=235,
                    color=color,
                    thickness=2,lineType=cv2.LINE_4)
        rectangle_width=40
        rectangle_height=20
        x1_rect=x_center-rectangle_width//2
        x2_rect=x_center+rectangle_width//2
        y1_rect=(y2-rectangle_height//2)+15
        y2_rect=(y2+rectangle_width//2)+15
        
        if track_id is not None:
            cv2.rectangle(frame,
                          int(x1_rect),int(y1_rect),
                          int(x2_rect),int(y2_rect),
                          color,cv2.FILLED)
            x1_text=x1_rect+12
            if track_id>99:
                x1_text-=10
            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text),int(y2_rect+5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,0,0),2
                        )
                
        
        return frame
    
    def draw_annotations(self,video_frames,tracks):
        output_frames=[]
        for frame_num,frame in enumerate(video_frames):
            frame=frame.copy()

            player_dict=tracks["players"][frame_num]
            ball_dict=tracks["ball"][frame_num]
            referee_dict=tracks["referees"][frame_num]
            
            #draw player bounding boxes
            for track_id,player in player_dict.items():
                frame=self.draw_ellipse(frame,player["bounding_box"],(0,255,0),track_id)
                # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            output_frames.append(frame)
            
        return output_frames