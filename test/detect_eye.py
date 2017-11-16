import cv2


#--------------------------------------------------------------------------
# load Classifiers
# return (faceCascade, noseCascade)
def load_Classifiers():
    baseCascadePath = "./model/"
    faceCascadeFilePath = baseCascadePath + "haarcascade_frontalface_default.xml"
    noseCascadeFilePath = baseCascadePath + "haarcascade_mcs_nose.xml"
    fistCascadeFilePath = baseCascadePath + "fist.xml"
    eyeCascadeFilePath = baseCascadePath + "haarcascade_eye.xml"

    # build our cv2 Cascade Classifiers
    faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
    noseCascade = cv2.CascadeClassifier(noseCascadeFilePath)
    fistCascade = cv2.CascadeClassifier(fistCascadeFilePath)
    eyeCascade = cv2.CascadeClassifier(eyeCascadeFilePath)

    return (faceCascade, noseCascade, fistCascade, eyeCascade)


if __name__ == "__main__":
    # load face, fist and nose Classifiers
    (faceCascade, noseCascade, fistCascade, eyeCascade) = load_Classifiers()

    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    while True:

        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)

        render = frame.copy()

        #-------------------------face------------------
        face = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30),
            #opencv version is 3.1
            flags = cv2.CASCADE_SCALE_IMAGE)
        # flags=cv2.cv.CV_HAAR_SCALE_IMAGE if the opencv version is 2

        face_top, face_bottom, face_left, face_right = 0, 0, 0, 0

        for x,y,w,h in face:
            cv2.rectangle(render,(x,y),(x+w,y+h),(0,0,255),2)

            roi_grey = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            print x, y, w, h

            face_top, face_bottom, face_left, face_right = y, y + h, x, x + w

            eyes = eyeCascade.detectMultiScale(roi_grey)
            for ex,ey,ew,eh in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex + ew, ey + eh),(0,255,0),2)


        cv2.imshow('frame',frame)

        if cv2.waitKey(1) &0xFF == ord('q'):
            break






    cap.release()
    cv2.destroyAllWindows()