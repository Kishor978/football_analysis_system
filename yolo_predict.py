from ultralytics import YOLO
model = YOLO("models\\best.pt")

results = model.predict('data/video1.mp4',save=True)

print(results[0])
for box in results[0].boxes:
    print(box)