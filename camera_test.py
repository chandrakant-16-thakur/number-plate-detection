import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not opened")
    exit()

ret, frame = cap.read()

print("Ret:", ret)

if ret:
    cv2.imwrite("camera_test.jpg", frame)
    print("Image Saved Successfully")
else:
    print("Camera Read Failed")

cap.release()