import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt') 
model.fuse() # Optimize model for inference

cap = cv2.VideoCapture(1)
fps = cap.get(cv2.CAP_PROP_FPS)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fourcc = cv2.VideoWriter.fourcc(*'mp4v')
# fourcc = cap.get(cv2.CAP_PROP_FOURCC)

print(fourcc)

out = cv2.VideoWriter(filename= "output.mp4", fourcc=fourcc, fps=int(fps), frameSize=(int(w), int(h)))

try:
    while True:
        ret, frame = cap.read()
        result = model(frame)[0]
        print(result)

        # Annotate with detections
        if len(result.boxes.xyxy) > 0:
            for box in result.boxes:
                xyxy = box.xyxy.cpu().numpy()[0]
                cls = int(box.cls.cpu().numpy()[0])
                conf = box.conf.cpu().numpy()[0]
                # print(box.cls, box.conf)
                frame = cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                # Add class name and confidence
                frame = cv2.putText(frame, f"{result.names[cls]} {conf:.2f}", (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
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
