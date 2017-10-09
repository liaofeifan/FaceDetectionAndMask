import cv2

# location of OpenCV Haar Cascade Classifiers:
root_path = "/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/"

# xml files describing our haar cascade classifiers
face_cascade_path = root_path + "haarcascade_frontalface_default.xml"
eye_cascade_path = root_path + "haarcascade_eye.xml"
# build our cv2 Cascade Classifiers
faceCascade = cv2.CascadeClassifier(face_cascade_path)
noseCascade = cv2.CascadeClassifier(eye_cascade_path)


cascade = faceCascade
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        #opencv version is 3.1
        flags = cv2.CASCADE_SCALE_IMAGE)
    # flags=cv2.cv.CV_HAAR_SCALE_IMAGE if the opencv version is 2
    for x,y,z,w in face:
        cv2.rectangle(frame,(x,y),(x+z,y+w),(0,0,255),2)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) &0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
