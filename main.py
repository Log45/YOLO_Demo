import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO('yolov8l.pt') 
model.fuse() 

box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture("roblox.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fourcc = cv2.VideoWriter.fourcc(*'mp4v')

out = cv2.VideoWriter(filename= "output.mp4", fourcc= fourcc, fps=30, frameSize=(1280, 720))

try:
    while True:
        ret, frame = cap.read()
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        frame = box_annotator.annotate(frame, detections)
        frame = label_annotator.annotate(frame, detections)

        out.write(frame)
        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break
        if cv2.waitKey(12) & 0xFF == ord('q'):
            break
except:
    pass
finally:
    print("Recording quit.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
