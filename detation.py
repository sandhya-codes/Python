import cv2  
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    blurred = cv2.GaussianBlur(frame, (35, 35), 0)
    for (x, y, w, h) in faces:
        blurred[y:y+h, x:x+w] = frame[y:y+h, x:x+w]  
        cv2.rectangle(blurred, (x, y), (x+w, y+h), (0, 255, 0), 2)   
    count = len(faces)
    cv2.putText(blurred, f"Faces Detected: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(blurred, "Press C to Capture | Q to Quit", (10, blurred.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imshow(" Face Detection App", blurred)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        cv2.imwrite("face_snapshot.png", frame)
        print("📸 Snapshot saved as 'face_snapshot.png'")

cam.release()
cv2.destroyAllWindows()
