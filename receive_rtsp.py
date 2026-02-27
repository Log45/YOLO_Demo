import cv2

def receive_rtsp(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print(f"Error opening video stream: {rtsp_url}")
        return
    
    retries = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if retries < 10:
                print("Failed to grab frame, retrying...")
                retries += 1
                continue
            print("Failed to grab frame")
            break
        
        cv2.imshow("RTSP Stream", frame)

        if (cv2.waitKey(30) == 27):
            break
        if cv2.waitKey(12) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    rtsp_url = "rtsp://localhost:8554/stream"
    receive_rtsp(rtsp_url)