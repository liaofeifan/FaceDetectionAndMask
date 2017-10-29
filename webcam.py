import cv2


# cascade = cv2.CascadeClassifier('/opt/opencv3/data/haarcascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('/opt/opencv3/data/haarcascades/haarcascade_eye.xml')
# capture = cv2.cvCreateCameraCapture(0)
#
# cv2.cvSetCaptureProperty(capture, cv2.CV_CAP_PROP_FRAME_WIDTH, 320)
# cv2.cvSetCaptureProperty(capture, cv2.CV_CAP_PROP_FRAME_HEIGHT, 240)

capset = cv2.VideoCapture(0)
capset.set(3, 1280)
capset.set(4, 1024)

print capset.get(3)






while True:

    ret,frame = capset.read()
    # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # face = cascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(30,30),
    #     #opencv version is 3.1
    #     flags = cv2.CASCADE_SCALE_IMAGE)
    # # flags=cv2.cv.CV_HAAR_SCALE_IMAGE if the opencv version is 2
    #
    # # here we detect eyes
    #
    # for x,y,w,h in face:
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    #
    #     roi_grey = gray[y:y + h, x:x + w]
    #     roi_color = frame[y:y + h, x:x + w]
    #
    #     eyes = eye_cascade.detectMultiScale(roi_grey)
    #     for ex,ey,ew,eh in eyes:
    #         cv2.rectangle(roi_color,(ex,ey),(ex + ew, ey + eh),(0,255,0),2)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) &0xFF == ord('q'):
        break






cap.release()
cv2.destroyAllWindows()
