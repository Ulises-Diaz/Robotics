from ultralytics import YOLO 

model = YOLO('models/best.pt')

results = model.predict('input_videos/nebraska-volleyball---you-have-to-see-this-insane-rally-between-nebraska--kentucky.mp4', save = True)

print(results[0])

print('--------------------------------')

for box in results[0].boxes:
    print(box)