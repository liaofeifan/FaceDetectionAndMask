import cv2
import imutils
import numpy as np


# --------------------------------------------------------------------------
# load Classifiers
def load_Classifiers():
    baseCascadePath = "./model/"
    faceCascadeFilePath = baseCascadePath + "haarcascade_frontalface_default.xml"
    noseCascadeFilePath = baseCascadePath + "haarcascade_mcs_nose.xml"

    # build our cv2 Cascade Classifiers
    faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
    noseCascade = cv2.CascadeClassifier(noseCascadeFilePath)

    return (faceCascade, noseCascade)




# --------------------------------------------------------------------------
# get fingers extreme points
def get_extreme_points(segmented):
    chull = cv2.convexHull(segmented)

    # chull = segmented

    # print chull
    # find the most extreme points in the convex hull
    extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right = tuple(chull[chull[:, :, 0].argmax()][0])

    # print extreme_top

    return extreme_top, extreme_bottom, extreme_left, extreme_right


# --------------------------------------------------------------------------
# calculate the distance between (x1, y1) and (x2, y2)
def cal_dis(x1, y1, x2, y2):
    return np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))




# --------------------------------------------------------------------------
# use ycrcb to segment the mask and get the finger contour
def segment_ycrcb(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    mask2 = cv2.inRange(ycrcb, np.array([54, 131, 110]), np.array([163, 157, 135]))

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


# --------------------------------------------------------------------------
# use hsv to segment the mask and get the finger contour
def segment_hsv(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([-25, 50, 40]), np.array([25, 153, 255]))

    # ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    #
    # mask2 = cv2.inRange(ycrcb, np.array([54, 131, 110]), np.array([163, 157, 135]))

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


# -----------------------------------------------------------------------------
# resize the nask to DecorationWidth and DecorationHeight
def resize_mask(imgDecoration, orig_mask, orig_mask_inv, DecorationWidth, DecorationHeight):
    resized_imgDecoration = cv2.resize(imgDecoration, (DecorationWidth, DecorationHeight), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(orig_mask, (DecorationWidth, DecorationHeight), interpolation=cv2.INTER_AREA)
    resized_mask_inv = cv2.resize(orig_mask_inv, (DecorationWidth, DecorationHeight), interpolation=cv2.INTER_AREA)

    return (resized_imgDecoration, resized_mask, resized_mask_inv)


# -----------------------------------------------------------------------------
# get the joined image(ROI), p.s. the imgDecoration, mask and mask_inv here should be resized if necessary
# add mask to according coordinate
def add_mask_to_ROI(imgDecoration, mask, mask_inv, roi_color):
    # (top,left)----------------------------(top,right)
    # (y1,x1)                                   (y1,x2)
    # -                                         -
    # -                                         -
    # -                                         -
    # -                                         -
    # -                                         -
    # -                                         -
    # -                                         -
    # -                                         -
    # -                                         -
    # -                                         -
    # -                                         -
    # (y2,x1)                                   (y2,x2)
    # (bottom,left)--------------------------(bottom,right)
    # take ROI for mustache from background equal to size of mustache image
    # roi = roi_color[top:bottom, left:right]
    roi = roi_color
    # roi_bg contains the original image only
    #  where the mustache is not
    # in the region that is the size of the mustache.
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # roi_fg contains the image of the mustache only where the mustache is
    roi_fg = cv2.bitwise_and(imgDecoration, imgDecoration, mask=mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg, roi_fg)

    return dst


# -----------------------------------------------------------------------------
# detect face and add mask to the face
def add_mask_to_face(gray, faceCascade, noseCascade, imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight,
                     origDecorationWidth):
    # Detect faces in input video stream
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    top = 0
    bottom = 0
    left = 0
    right = 0

    # Iterate over each face found
    for (x, y, w, h) in faces:
        # Un-comment the next line for debug (draw box around all faces)
        # face = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        faceWidth = w
        faceHeight = h

        top = y1
        bottom = y2
        left = x1
        right = x2

        roi_gray = gray[y1:y2, x1:x2]
        roi_color = frame[y1:y2, x1:x2]

        # Detect a nose within the region bounded by each face (the ROI)
        nose = noseCascade.detectMultiScale(roi_gray)

        for (nx, ny, nw, nh) in nose:
            # Un-comment the next line for debug (draw box around the nose)
            # cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)

            # The mustache should be three times the width of the nose
            DecorationWidth = int(round(5 * faceWidth))
            DecorationHeight = DecorationWidth * origDecorationHeight / origDecorationWidth

            # Center the mustache on the bottom of the nose
            x1 = nx - (DecorationWidth / 3)
            x2 = nx + nw + (DecorationWidth / 3)
            y1 = ny + nh - (DecorationHeight / 2)
            y2 = ny + nh + (DecorationHeight / 2)

            # Check for clipping
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h

            # Re-calculate the width and height of the mustache image
            DecorationWidth = x2 - x1
            DecorationHeight = y2 - y1

            # Re-size the original image and the masks to the mustache sizes
            # calcualted above
            (resized_imgDecoration, resized_mask, resized_mask_inv) = resize_mask(imgDecoration, orig_mask,
                                                                                  orig_mask_inv, DecorationWidth,
                                                                                  DecorationHeight)

            # get the joined image, saved to dst back over the original image
            dst = add_mask_to_ROI(resized_imgDecoration, resized_mask, resized_mask_inv, roi_color)

            # place the joined image, saved to dst back over the original image
            roi_color[y1:y2, x1:x2] = dst
            break

    return frame, top, bottom, left, right


# -----------------------------------------------------------------------------
# dihibition_mask
def exhibit_mask(frame, mask_dictionary):
    # get the frame height and width
    (frame_height, frame_width) = frame.shape[:2]

    # the number of masks which need to be exhibit
    mask_num = len(mask_dictionary)

    count = 0
    for mask_name, mask_status in mask_dictionary.iteritems():

        if (mask_status == 1):
            # -----load and display mask ---
            # load masks
            (imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth) = load_masks(
                mask_name)

            # draw the exhibition area for the masks
            top, bottom, left, right = int(0.4 * frame_height) * count, int((0.3 + count * 0.3) * frame_height), 0, int(
                0.2 * frame_width)

            # resize the mask1
            DecorationWidth = right - left
            DecorationHeight = bottom - top

            (resized_imgDecoration, resized_mask, resized_mask_inv) = resize_mask(imgDecoration, orig_mask,
                                                                                  orig_mask_inv, DecorationWidth,
                                                                                  DecorationHeight)

            roi_color = frame[top:bottom, left:right]
            dst = add_mask_to_ROI(resized_imgDecoration, resized_mask, resized_mask_inv, roi_color)
            frame[top:bottom, left:right] = dst

            count += 1

    return frame


# -----------------------------------------------------------------------------
# finger detection and counting
def finger_detect(frame, top, right, bottom, left):
    # resize the frame
    frame = imutils.resize(frame, width=700)

    # flip the frame so that it is not the mirror view ????????????
    # frame = cv2.flip(frame, 1)


    # get the height and width of the frame
    (height, width) = frame.shape[:2]

    # get the ROI
    roi = frame[top:bottom, right:left]

    # convert the roi to grayscale and blur it
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # to get the background, keep looking till a threshold is read
    # so that our running average models gets calibrated

    # segment the hand region
    hand = segment_hsv(roi)

    extreme_top = 0

    # check whether hand region is segmented
    if hand is not None:
        # if yes, up[ack the threshold image and segmented region
        (thresholded, segmented) = hand

        # draw the semgent retion nd display the frame****
        cv2.drawContours(frame, [segmented + (right, top)], -1, (0, 0, 255))
        # cv2.imshow("Thresholded", thresholded)

        extreme_top, extreme_bottom, extreme_left, extreme_right = get_extreme_points(segmented)

    # draw the segmented hand
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

    return frame, extreme_top


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # collect video input from first webcam on system
    video_capture = cv2.VideoCapture(0)

    video_capture.set(3, 320)
    video_capture.set(4, 240)

    print video_capture.get(3)


    # the coordinate of detect face
    face_top, face_bottom, face_left, face_right = 0, 0, 0, 0

    # mask_dictionary {"mask_name": mask_status}
    # status 1 : display
    # status 2 : drag
    # status 3 : add on face
    mask_dict = {"res/ironman.png": 1, "res/mustache.png": 1}

    while True:

        # --------------------phase 0: read the frame and gray scale frame--------------------------------
        # Capture video feed
        ret, frame = video_capture.read()

        # Create greyscale image from the video feed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --------------------phase 1: display all the masks at the left side of the frame----------------
        frame = exhibit_mask(frame, mask_dict)


        # load face and nose Classifiers
        (faceCascade, noseCascade) = load_Classifiers()







        cv2.imshow('Video', frame)

        # press any key to exit
        # NOTE;  x86 systems may need to remove: " 0xFF == ord('q')"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()