import cv2
import numpy as np


# get fingers extreme points
def get_extreme_points(thresholded, segmented):
    chull = cv2.convexHull(segmented)

    # chull = segmented


    # print chull
    # find the most extreme points in the convex hull
    extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right = tuple(chull[chull[:, :, 0].argmax()][0])

    print extreme_top

    return extreme_top, extreme_bottom, extreme_left, extreme_right




def segment_hsv(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([-25, 50, 40]), np.array([25, 153, 255]))


    # Kernel matrices for morphological transformation
    kernel_square = np.ones((11, 11), np.uint8)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Perform morphological transformations to filter out the background noise
    # Dilation increase skin color area
    # Erosion increase skin color area
    dilation = cv2.dilate(mask2, kernel_ellipse, iterations=1)
    erosion = cv2.erode(dilation, kernel_square, iterations=1)
    dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)
    filtered = cv2.medianBlur(dilation2, 5)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    # ??????
    median = cv2.medianBlur(dilation2, 5)

    # thresh = cv2.threshold(median, 127, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thresh = cv2.threshold(median, 127, 255, cv2.THRESH_BINARY)[1]


    # Find contours of the filtered frame
    _, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)



    if len(contours) == 0:
        return
    else:
        segmented = max(contours, key=cv2.contourArea)
        return (thresh, segmented)



# --------------------------------------------------------------------------
# use ycrcb to segment the mask and get the finger contour
def segment_ycrcb(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    mask2 = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))

    cv2.imshow("show", mask2)
    # Kernel matrices for morphological transformation
    kernel_square = np.ones((11, 11), np.uint8)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Perform morphological transformations to filter out the background noise
    # Dilation increase skin color area
    # Erosion increase skin color area
    dilation = cv2.dilate(mask2, kernel_ellipse, iterations=1)
    erosion = cv2.erode(dilation, kernel_square, iterations=1)
    dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)
    filtered = cv2.medianBlur(dilation2, 5)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    # ??????
    median = cv2.medianBlur(dilation2, 5)

    thresh = cv2.threshold(median, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # thresh = cv2.threshold(median, 127, 255, cv2.THRESH_BINARY)[1]


    # Find contours of the filtered frame
    _, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return
    else:
        segmented = max(contours, key=cv2.contourArea)
        return (thresh, segmented)



if __name__ == "__main__":

    cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('model/haarcascade_eye.xml')


    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    print cap.get(3)

    while True:

        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        render = frame.copy()

        #-------------------------face------------------
        face = cascade.detectMultiScale(
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

            eyes = eye_cascade.detectMultiScale(roi_grey)
            for ex,ey,ew,eh in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex + ew, ey + eh),(0,255,0),2)

        neck_height = 30
        face_bottom += neck_height



        width = face_right - face_left
        height = face_bottom - face_top

        if width is not 0 and height is not 0:


            print width, height
            frame_roi = frame[face_top:face_bottom, face_left:face_right]
            white_mask = np.zeros((height+2, width+2), np.uint8)
            cv2.floodFill(frame_roi, white_mask, (0, 0), (255,255,255))

            frame_roi = cv2.bitwise_not(frame_roi)

            frame[face_top:face_bottom, face_left:face_right] = frame_roi

        #---------------------------finger--------------

        hand = segment_ycrcb(frame)

        # check whether hand region is segmented
        if hand is not None:
            # if yes, up[ack the threshold image and segmented region
            (thresholded, segmented) = hand

            cv2.drawContours(render, segmented, -1, (0, 0, 255), 3)

            # cv2.imshow("Thresholded", thresholded)

            # count the number of fingers
            extreme_top, extreme_bottom, extreme_left, extreme_right = get_extreme_points(thresholded, segmented)
            #
            #
            #
            cv2.circle(render, extreme_top, 8, (0, 0, 255), -1)



        # cv2.imshow('frame',render)

        if cv2.waitKey(1) &0xFF == ord('q'):
            break






    cap.release()
    cv2.destroyAllWindows()
