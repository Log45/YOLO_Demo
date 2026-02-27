import cv2
from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt', "pose")
model.fuse()

cap = cv2.VideoCapture(1)
fps = cap.get(cv2.CAP_PROP_FPS)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fourcc = cv2.VideoWriter.fourcc(*'mp4v')
out = cv2.VideoWriter(filename= "output.mp4", fourcc=fourcc, fps=int(fps), frameSize=(int(w), int(h)))

try:
    while True:
        ret, frame = cap.read()
        result = model(frame)[0]
        # print(result)
        if result.keypoints is not None:
            # print(result.keypoints)
            if len(result.keypoints.xy.cpu().numpy()) > 0:
                # i = 0
                # while i < len(result.keypoints.xy.cpu().numpy()[0])-1:
                #     print("Keypoint:", i)
                #     p1 = result.keypoints.xy.cpu().numpy()[0][i]
                #     p2 = result.keypoints.xy.cpu().numpy()[0][i+1]
                #     print(p1, p2)
                #     frame = cv2.circle(frame, (int(p1[0]), int(p1[1])), 8, (255, 0, 0), 5)
                #     frame = cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 5)
                #     i += 1
                for p in result.keypoints.xy.cpu().numpy()[0]:
                    # print(p)
                    frame = cv2.circle(frame, (int(p[0]), int(p[1])), 5, (255, 0, 0), 5)


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
        cv2.imshow("yolov11-pose", frame)

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