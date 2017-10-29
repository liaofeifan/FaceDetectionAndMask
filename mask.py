import cv2
import imutils
import numpy as np

# global variables
bg = None


#--------------------------------------------------------------------------
# load Classifiers
def load_Classifiers():
    baseCascadePath = "./model/"
    faceCascadeFilePath = baseCascadePath + "haarcascade_frontalface_default.xml"
    noseCascadeFilePath = baseCascadePath + "haarcascade_mcs_nose.xml"

    # build our cv2 Cascade Classifiers
    faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
    noseCascade = cv2.CascadeClassifier(noseCascadeFilePath)

    return (faceCascade, noseCascade)


#--------------------------------------------------------------------------
# Load our overlay image and compact the paramatars
def load_masks(name):
    imgDecoration = cv2.imread(name, -1)

    # Create the mask for the mustache
    orig_mask = imgDecoration[:, :, 3]

    # Create the inverted mask for the mustache
    orig_mask_inv = cv2.bitwise_not(orig_mask)

    # Convert mustache image to BGR
    # and save the original image size (used later when re-sizing the image)
    imgDecoration = imgDecoration[:, :, 0:3]
    origDecorationHeight, origDecorationWidth = imgDecoration.shape[:2]

    return (imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth)


#--------------------------------------------------------------------------
# To find the running average over the backgound
def run_avg(image, aWeight):
    global bg
    # initialize the backgound
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accmulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)


#--------------------------------------------------------------------------
# To segment the region of hand in the image with threshold value method
def segment_thr(image, threshold = 10):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"),image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the threshold image
    (_, cnts, _) = cv2. findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detectedimgDecoration,
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key = cv2.contourArea)
        return (thresholded, segmented)

# --------------------------------------------------------------------------
# To segment the region of hand in the image with blur method
# https://digibee.co.in/2017/05/24/gesture-recognition-using-open-cv-and-python/
# Here 127 is the threshold value.The white area have a pixel value less than 127
# and the black portion have a pixel value greater than 127.
# we update change this value according to the light and background condition
def segment_blur(grey_image, thr = 127):
    blurred = cv2.GaussianBlur(grey_image, (35,35),0)

    threshold_value = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    _, contours, hierarchy = cv2.findContours(threshold_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    segmented = max(contours, key=cv2.contourArea)
    return (threshold_value, segmented)


# --------------------------------------------------------------------------
# hand finger counting
from sklearn.metrics import pairwise
def count(thresholded, segmented):
    # find the convex hull of the segmented hand region
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the palm
    cX = (extreme_left[0] + extreme_right[0]) / 2
    cY = (extreme_top[1] + extreme_bottom[1]) / 2

    # find the maximum eucidean distance between the center of the palm
    # and the most extreme pints of the convex hull
    distance = pairwise.euclidean_distances([(cX, cY)], Y = [extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    # calculate the radius of the circle with 80% ot the max eucidean
    radius = int(0.8 * maximum_distance)

    # find the circumference of the circle
    circumference = (2 * np.pi * radius)

    # take out the circular ROI which has the palm and the fingers
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")

    #draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # take bit-wise AND between thresholed hand using the circular ROI
    # whcih gives the cuts obtained using mask on the thresholded hand
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # compute the contours in the circular ROI
    (_, cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # initialize the the finger count
    count = 0

    # loop through the contours found
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # increment the count of fingers only if
        # 1. the contour region is not the wrist (bottom area)
        # 2. the number of points along the contour does not exceed 25% of the circumference of the circular ROI
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count

#-----------------------------------------------------------------------------
# detect face and add mask to the face


#-----------------------------------------------------------------------------
# resize the nask to DecorationWidth and DecorationHeight
def resize_mask(imgDecoration, orig_mask, orig_mask_inv,  DecorationWidth,DecorationHeight):
    resized_imgDecoration = cv2.resize(imgDecoration, (DecorationWidth, DecorationHeight), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(orig_mask, (DecorationWidth, DecorationHeight), interpolation=cv2.INTER_AREA)
    resized_mask_inv = cv2.resize(orig_mask_inv, (DecorationWidth, DecorationHeight), interpolation=cv2.INTER_AREA)

    return (resized_imgDecoration, resized_mask, resized_mask_inv)

#-----------------------------------------------------------------------------
# get the joined image(ROI), p.s. the imgDecoration, mask and mask_inv here should be resized if necessary
# add mask to according coordinate
def add_mask_to_ROI(top, bottom, left, right, imgDecoration, mask, mask_inv, roi_color):
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


#-----------------------------------------------------------------------------
# add mask to face
def add_mask_to_face(gray, faceCascade, noseCascade, imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth):
    # Detect faces in input video stream
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

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
            dst = add_mask_to_ROI(y1, y2, x1, x2, resized_imgDecoration, resized_mask, resized_mask_inv, roi_color)

            # place the joined image, saved to dst back over the original image
            roi_color[y1:y2, x1:x2] = dst
            break

    return frame


#-----------------------------------------------------------------------------
# dihibition_mask
def exhibit_mask(frame):
    # get the frame height and width
    (frame_height, frame_width) = frame.shape[:2]

    # -----load and display mask 1---
    # load masks
    (imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth) = load_masks("res/ironman.png")


    # draw the exhibition area for the masks
    top, bottom, left, right = 0, int(0.3 *frame_height), 0, int(0.2 * frame_width)

    #resize the mask1
    DecorationWidth = right - left
    DecorationHeight = bottom - top

    (resized_imgDecoration, resized_mask, resized_mask_inv) = resize_mask(imgDecoration, orig_mask, orig_mask_inv, DecorationWidth, DecorationHeight)

    roi_color = frame[top:bottom,left:right]
    dst = add_mask_to_ROI(top,bottom,left,right,resized_imgDecoration,resized_mask,resized_mask_inv,roi_color)
    frame[top:bottom,left:right] = dst

    # -----load and display mask 2---
    #load masks
    (imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth) = load_masks("res/mustache.png")

    # draw the exhibition area for the masks
    top, bottom, left, right = int(0.4 * frame_height), int(0.6 * frame_height), 0, int(0.2 * frame_width)
    # top, bottom, left, right = 2, int(0.3 * frame_height) + bottom, 0, int(0.2 * frame_width)

    # resize the mask1
    DecorationWidth = right - left
    DecorationHeight = bottom - top

    (resized_imgDecoration, resized_mask, resized_mask_inv) = resize_mask(imgDecoration, orig_mask, orig_mask_inv, DecorationWidth, DecorationHeight)

    roi_color = frame[top:bottom, left:right]
    dst = add_mask_to_ROI(top, bottom, left, right, resized_imgDecoration, resized_mask, resized_mask_inv, roi_color)
    frame[top:bottom, left:right] = dst


    return frame


#-----------------------------------------------------------------------------
# finger detection and counting
def finger_detect_and_count(frame, aWeight,num_frames, top, right, bottom, left):
    # resize the frame
    frame = imutils.resize(frame, width=700)

    # flip the frame so that it is not the mirror view ????????????
    # frame = cv2.flip(frame, 1)

    # clone the crmae*******
    clone = frame.copy()

    # get the height and width of the frame
    (height, width) = frame.shape[:2]

    # get the ROI
    roi = frame[top:bottom, right:left]

    # convert the roi to grayscale and blur it
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # to get the background, keep looking till a threshold is read
    # so that our running average models gets calibrated
    if num_frames < 30:
        run_avg(gray, aWeight)
    else:
        # segment the hand region
        hand = segment_blur(gray)

        # check whether hand region is segmented
        if hand is not None:
            # if yes, up[ack the threshold image and segmented region
            (thresholded, segmented) = hand

            # draw the semgent retion nd display the frame****
            cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
            # cv2.imshow("Thresholded", thresholded)

            # count the number of fingers
            fingers = count(thresholded, segmented)

            cv2.putText(clone, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # draw the segmented hand
    cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0))

    return clone




if __name__ == "__main__":
    #-----------------------------------------------------------------------------
    # collect video input from first webcam on system


    video_capture = cv2.VideoCapture(0)

    video_capture.set(3,320)
    video_capture.set(4,240)



    print video_capture.get(3)

    # initialize weight
    aWeight = 0.5

    # ROI coordinateds
    finger_detect_rec_top, finger_detect_rec_right, finger_detect_rec_bottom, finger_detect_rec_left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0

    flag = True

    while True:

        if not flag:
            flag = not flag
            continue
        #--------------------phase 0: read the frame and gray scale frame--------------------------------
        # Capture video feed
        ret, frame = video_capture.read()

        # Create greyscale image from the video feed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --------------------phase 1: display all the masks at the left side of the frame----------------
        frame = exhibit_mask(frame)

        # --------------------phase 2: add mask to face-------------------
        # load face and nose Classifiers
        (faceCascade, noseCascade) = load_Classifiers()

        # load masks
        (imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth) = load_masks("res/ironman.png")

        # add mask to face
        frame = add_mask_to_face(gray, faceCascade, noseCascade, imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth)

        # ---------------------phase 3: finger detection-------------------------
        frame = finger_detect_and_count(frame,aWeight,num_frames,finger_detect_rec_top, finger_detect_rec_right, finger_detect_rec_bottom, finger_detect_rec_left)
        num_frames += 1
        # ---------------------phase 4: hook the finger and the mask-------------
        # ---------------------phase 5: drag the mask to face--------------------
        # ---------------------phase 6: overlay mask to face---------------------
        # Display the resulting frame
        cv2.imshow('Video', frame)

        # press any key to exit
        # NOTE;  x86 systems may need to remove: " 0xFF == ord('q')"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()