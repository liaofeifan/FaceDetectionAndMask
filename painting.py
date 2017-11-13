import cv2
import numpy as np

stop_time = 100


#--------------------------------------------------------------------------
# load Classifiers
# return (faceCascade, noseCascade)
def load_Classifiers():
    baseCascadePath = "./model/"
    faceCascadeFilePath = baseCascadePath + "haarcascade_frontalface_default.xml"
    noseCascadeFilePath = baseCascadePath + "haarcascade_mcs_nose.xml"

    # build our cv2 Cascade Classifiers
    faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
    noseCascade = cv2.CascadeClassifier(noseCascadeFilePath)

    return (faceCascade, noseCascade)


# --------------------------------------------------------------------------
# use hsv to segment the mask and get the finger contour
# i prefer to use HSV when in the classroom
# return (thresh, segmented) or none if not detected
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

    thresh = cv2.threshold(median, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # thresh = cv2.threshold(median, 127, 255, cv2.THRESH_BINARY)[1]


    # Find contours of the filtered frame
    _, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]

    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)


    if len(contours) == 0:
        return
    else:
        segmented = max(contours, key=cv2.contourArea)
    return (thresh, segmented, cnt, hull, defects)




#-----------------------------------------------------------------------------
# detect face (one face)
# return face_coor and nose_coor
# face_coor = [face_top, face_bottom, face_Left, face_right]
# nose_coor = [nose_top, nose_bottom, nose_left, nose_right]
def detect_face_and_nose(frame, render, faceCascade, noseCascade):

    # Create greyscale image from the video feed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in input video stream
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return ([0,0,0,0], [0,0,0,0])


    (x, y, w, h) = faces[0]

    # Un-comment the next line for debug (draw box around all faces)
    # face = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    faceWidth = w
    faceHeight = h

    face_top = y1
    face_bottom = y2
    face_left = x1
    face_right = x2

    face_coor = [face_top, face_bottom, face_left, face_right]

    roi_gray = gray[y1:y2, x1:x2]


    # draw face rect
    cv2.rectangle(render, (face_left, face_top), (face_right, face_bottom), (0, 0, 255), 2)

    nose_coor = [0, 0, 0, 0]

    # detect nose
    # for not I don't need to use it
    # nose = noseCascade.detectMultiScale(roi_gray)
    # if len(nose) == 0:
    #     return (face_coor, [0,0,0,0])
    #
    # (nx, ny, nw, nh) = nose[0]
    #
    # nose_top = ny
    # nose_bottom = ny + nh
    # nose_left = nx
    # nose_right = nx + nw
    #
    #
    # nose_coor = [nose_top, nose_bottom, nose_left, nose_right]

    return face_coor, nose_coor



#-----------------------------------------------------------------------------
# finger detection and get extreme_top
# return render and extreme_top (but only extreme_top should be used)
def detect_finger(frame, render, face_coor):
    top, bottom, left, right = face_coor[0], face_coor[1], face_coor[2], face_coor[3]
    # flip the frame so that it is not the mirror view ????????????
    # frame = cv2.flip(frame, 1)


    # get the height and width of the frame
    (frame_height, frame_width) = frame.shape[:2]

    # flood fill the face and neck region, so we can remove face region when detect fingers
    neck_height = 20
    bottom += neck_height


    if (bottom + 2) > frame_height:
        bottom = frame_height - 3

    width = right - left
    height = bottom - top

    if width is not 0 and height is not 0:

        frame_roi = frame[top:bottom, left:right]
        white_mask = np.zeros((height + 2, width + 2), np.uint8)
        cv2.floodFill(frame_roi, white_mask, (0, 0), (255, 255, 255))

        frame_roi = cv2.bitwise_not(frame_roi)

        frame[top:bottom, left:right] = frame_roi

    # to get the background, keep looking till a threshold is read
    # so that our running average models gets calibrated

    # segment the hand region
    hand = segment_hsv(frame)

    extreme_top = [0,0]

    # check whether hand region is segmented
    if hand is not None:
        # if yes, up[ack the threshold image and segmented region
        (thresholded, segmented, cnt, hull, defects) = hand

        # draw the semgent retion nd display the frame****
        cv2.drawContours(render, segmented, -1, (0, 255, 255), 2)
        # cv2.imshow("Thresholded", thresholded)

        extreme_top, extreme_bottom, extreme_left, extreme_right = get_extreme_points(segmented)

        cv2.circle(render, extreme_top, 8, (0, 0, 255), -1)

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                cv2.line(render, start, end, [0, 255, 0], 2)
                cv2.circle(render, far, 5, [0, 0, 255], -1)

    return render, extreme_top

#--------------------------------------------------------------------------
# get fingers extreme points
def get_extreme_points(segmented):
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right = tuple(chull[chull[:, :, 0].argmax()][0])

    return extreme_top, extreme_bottom, extreme_left, extreme_right


def painting(frame, render, pts):
    cv2.polylines(render, [np.array(pts)], False, (255, 0, 0), 1)
    return render



if __name__ == "__main__":

    #-----------------------------------------------------------------------------
    # init
    #-----------------------------------------------------------------------------
    video_capture = cv2.VideoCapture(0)

    # set video resolution
    video_capture.set(3,320)
    video_capture.set(4,240)

    (faceCascade, noseCascade) = load_Classifiers()

    # count how many frames drag be shadowed
    state_4_count = 0

    finger_print = []

    stop_count = 0

    while True:

        # -----------------------------------------------------------------------------
        # phase 0: load frame
        # -----------------------------------------------------------------------------

        # Capture video feed
        ret, frame = video_capture.read()
        # we use frame to detect and segment
        # and use render to render image and imshow
        render = frame.copy()

        # -----------------------------------------------------------------------------
        # phase 1: exhibit masks on the left top of the display
        # -----------------------------------------------------------------------------

        # exhibit_masks(frame, render, mask_status, mask_coors, mask_info)

        # -----------------------------------------------------------------------------
        # phase 2: detect face and nose
        # -----------------------------------------------------------------------------

        (face_coor, nose_coor) = detect_face_and_nose(frame, render, faceCascade, noseCascade)


        # -----------------------------------------------------------------------------
        # phase 3: detect fingers
        # -----------------------------------------------------------------------------

        (render, extreme_top) = detect_finger(frame, render, face_coor)



        finger_print.append(extreme_top)
        painting(frame, render, finger_print)

        # imshow the frame
        cv2.imshow('Video', render)

        # press any key to exit
        # NOTE;  x86 systems may need to remove: " 0xFF == ord('q')"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()