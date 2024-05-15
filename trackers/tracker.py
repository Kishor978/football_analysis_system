from ultralytics import YOLO
import supervision as sv
import pickle
import os

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
        
    def get_object_tracks(self,frames,read_from_stubs=False,stub_path=None):
        if read_from_stubs and os.path.exists(stub_path) and stub_path is not None:
            with open(stub_path,"rb") as f:
                tracks=pickle.load(f)
            return tracks
        detections=self.detect_frames(frames)
        tracks={
            "player":[],
            "ball":[],
            "refree":[],
        }
        
        for frame_num,detection in enumerate(detections):
            cls_names=detection.names
            cls_names_inv={v:k for k,v in cls_names.items()}
            
            #convert to supervision format
            detection_supervision=sv.Detections.from_ultralytics(detection)
            
            #convert goalkeeper to player object
            for object_id,class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id]=='goalkeeper':
                    detection_supervision.class_id[object_id]=cls_names_inv['player']
                    
            #track objects
            detection_with_tracks=self.tracker.update_with_detections(detection_supervision)
            tracks["player"].append({})
            tracks["ball"].append({})
            tracks["refree"].append({})
            for frame_detection in detection_with_tracks:
                bounding_boxes=frame_detection[0].tolist()
                cls_id=frame_detection[3]
                track_id=frame_detection[4]
                if cls_id==cls_names_inv['player']:
                    tracks["player"][frame_num][track_id]={"bounding_box":bounding_boxes}
                if cls_id==cls_names_inv['refree']:
                    tracks["refree"][frame_num][track_id]={"bounding_box":bounding_boxes}   
            for frame_detection in detection_with_tracks:
                bounding_boxes=frame_detection[0].tolist()
                cls_id=frame_detection[3]
                track_id=frame_detection[4]
                if cls_id==cls_names_inv['ball']:
                    tracks["ball"][frame_num][1]={"bounding_box":bounding_boxes}
        if stub_path is not None:
            with open(stub_path,"wb") as f:
                pickle.dump(tracks,f)
        return tracks