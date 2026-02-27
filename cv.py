import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    cv2.imshow("frame", frame)

    if (cv2.waitKey(30) == 27):
        break
    if cv2.waitKey(12) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()