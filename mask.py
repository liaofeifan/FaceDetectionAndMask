import cv2
import imutils
import numpy as np

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

# --------------------------------------------------------------------------
# load all the mask info
def initialize_mask_info(mask_load_info):
    for mask_name, mask_info in mask_load_info.iteritems():
        (imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth) = load_masks(mask_name)
        mask_info[0], mask_info[1], mask_info[2] , mask_info[3], mask_info[4] = imgDecoration, \
                                                                                orig_mask, \
                                                                                orig_mask_inv, \
                                                                                origDecorationHeight, \
                                                                                origDecorationWidth
    return mask_load_info

# --------------------------------------------------------------------------
# initialize the coordinate of masks
def initialize_mask_dict(mask_dictionary, frame_height, frame_width):
    count = 0
    for mask_name, mask_status in mask_dictionary.iteritems():
        # draw the exhibition area for the masks
        top, bottom, left, right = int(0.4 * frame_height) * count, int((0.3 + count * 0.3) * frame_height), 0, int(
            0.2 * frame_width)

        mask_status[1] = top
        mask_status[2] = bottom
        mask_status[3] = left
        mask_status[4] = right

        count += 1



#--------------------------------------------------------------------------
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


#--------------------------------------------------------------------------
# calculate the distance between (x1, y1) and (x2, y2)
def cal_dis(x1, y1, x2, y2):
    return np.sqrt( np.power(x1 - x2, 2) + np.power(y1 - y2, 2) )


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
# use ycrcb to segment the mask and get the finger contour
def segment_ycrcb(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    mask2 = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))

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
    mask2 = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))

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



#-----------------------------------------------------------------------------
# detect face and add mask to the face
def add_mask_to_face(render, gray, faceCascade, noseCascade, mask_dictionary, mask_load_info):
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
        roi_color = render[y1:y2, x1:x2]

        cv2.rectangle(render, (left, top), (right, bottom), (0, 0, 255), 2)


        # find if this mask need to be add to face
        for mask_name, mask_status in mask_dictionary.iteritems():

            if mask_status[0] == 3:
                # load masks
                (imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth) = \
                            mask_load_info[mask_name][0], \
                            mask_load_info[mask_name][1], \
                            mask_load_info[mask_name][2], \
                            mask_load_info[mask_name][3], \
                            mask_load_info[mask_name][4]

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

    return render, top, bottom, left, right


#-----------------------------------------------------------------------------
# dihibition_mask
def exhibit_mask(frame, render, mask_dictionary, mask_load_info):
    # get the frame height and width
    (frame_height, frame_width) = frame.shape[:2]


    # the number of masks which need to be exhibit
    mask_num = len(mask_dictionary)

    count = 0
    for mask_name, mask_status in mask_dictionary.iteritems():

        if (mask_status[0] == 1):

            # -----load and display mask ---
            # load masks


            (imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth) = mask_load_info[mask_name][0], \
                                                                                                   mask_load_info[mask_name][1], \
                                                                                                   mask_load_info[mask_name][2], \
                                                                                                   mask_load_info[mask_name][3], \
                                                                                                   mask_load_info[mask_name][4]

            # # draw the exhibition area for the masks
            # top, bottom, left, right = int(0.4 * frame_height) * count, int( (0.3 + count * 0.3) * frame_height), 0, int(0.2 * frame_width)

            # mask_status[1] = top
            # mask_status[2] = bottom
            # mask_status[3] = left
            # mask_status[4] = right

            top = mask_status[1]
            bottom = mask_status[2]
            left = mask_status[3]
            right = mask_status[4]


            # resize the mask1
            DecorationWidth = right - left
            DecorationHeight = bottom - top

            (resized_imgDecoration, resized_mask, resized_mask_inv) = resize_mask(imgDecoration, orig_mask,
                                                                                  orig_mask_inv, DecorationWidth,
                                                                                  DecorationHeight)

            roi_color = render[top:bottom, left:right]
            dst = add_mask_to_ROI(resized_imgDecoration, resized_mask, resized_mask_inv, roi_color)
            render[top:bottom, left:right] = dst


            count += 1


    return render


#-----------------------------------------------------------------------------
# finger detection and counting
def finger_detect(frame, render, top,  bottom, left, right):


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
    hand = segment_ycrcb(frame)

    extreme_top = 0

    # check whether hand region is segmented
    if hand is not None:
        # if yes, up[ack the threshold image and segmented region
        (thresholded, segmented) = hand

        # draw the semgent retion nd display the frame****
        cv2.drawContours(render, segmented, -1, (0, 255, 255), 2)
        # cv2.imshow("Thresholded", thresholded)

        extreme_top, extreme_bottom, extreme_left, extreme_right = get_extreme_points(segmented)

        cv2.circle(render, extreme_top, 8, (0, 0, 255), -1)

    return render, extreme_top



#-----------------------------------------------------------------------------
# drag mask with finger extreme top point
def drag_mask(frame, render, extreme_top, mask_dictionary, mask_load_info):

    # get the frame height and width
    (frame_height, frame_width) = frame.shape[:2]


    # check if already hooked
    for mask_status in mask_dictionary.items():
        if mask_status[0] == 2:
            # a mask is been dragged
            return render

    # no mask is been dragged
    for mask_name, mask_status in mask_dictionary.iteritems():

        if mask_status[0] == 1:

            mask_exh_top = mask_status[1]
            mask_exh_bottom = mask_status[2]
            mask_exh_left = mask_status[3]
            mask_exh_right = mask_status[4]

            # resize the mask
            DecorationWidth = mask_exh_right - mask_exh_left
            DecorationHeight = mask_exh_bottom - mask_exh_top

            pic_exh_center = ( (mask_exh_left + mask_exh_right) / 2, (mask_exh_top + mask_exh_bottom) / 2 )

            dis_to_pic = cal_dis(extreme_top[0], extreme_top[1], pic_exh_center[0], pic_exh_center[1])


            if dis_to_pic < 10:

                mask_status[0] = 2
                print "find it"

                # laod mask info that needed to be dragged
                (imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth) = \
                mask_load_info[mask_name][0], \
                mask_load_info[mask_name][1], \
                mask_load_info[mask_name][2], \
                mask_load_info[mask_name][3], \
                mask_load_info[mask_name][4]

                # resize the dragging mask
                (resized_imgDecoration, resized_mask, resized_mask_inv) = resize_mask(imgDecoration, orig_mask,
                                                                                      orig_mask_inv,
                                                                                      DecorationWidth, DecorationHeight)

                top_mask, bottom_mask, left_mask, right_mask = extreme_top[1] - (DecorationHeight) / 2, \
                                                               extreme_top[1] + (DecorationHeight) / 2 + 1, \
                                                               extreme_top[0] - (DecorationWidth) / 2, \
                                                               extreme_top[0] + (DecorationWidth) / 2
                if top_mask < 0 or bottom_mask > frame_height or left_mask < 0 or right_mask > frame_width:
                    return render

                roi_mask = render[top_mask:bottom_mask, left_mask:right_mask]
                dst = add_mask_to_ROI(resized_imgDecoration, resized_mask, resized_mask_inv, roi_mask)
                # add the mask to frame
                render[top_mask:bottom_mask, left_mask:right_mask] = dst


    return render




if __name__ == "__main__":
    #-----------------------------------------------------------------------------
    # collect video input from first webcam on system
    video_capture = cv2.VideoCapture(0)

    video_capture.set(3,640)
    video_capture.set(4,480)



    print video_capture.get(3)


    # ROI coordinateds
    finger_detect_rec_top, finger_detect_rec_right, finger_detect_rec_bottom, finger_detect_rec_left = 10, 350, 225, 590

    # the coordinate of detect face
    face_top, face_bottom, face_left, face_right = 0, 0, 0, 0

    # mask_dictionary {"mask_name": [mask_status, top, bottom, left, right]}
    # status 1 : display
    # status 2 : drag
    # status 3 : add on face
    mask_dict = {"res/ironman.png" : [1, 0, 0, 0, 0], "res/mustache.png" : [1, 0, 0, 0, 0] }

    # initialize the coordinate of masks
    initialize_mask_dict(mask_dict, video_capture.get(4), video_capture.get(3))

    # load mask info in
    #  {"mask_name": [imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth]}
    mask_load_info = {"res/ironman.png" : [0,0,0,0,0], "res/mustache.png" : [0,0,0,0,0]}

    # load all the mask info
    mask_load_info = initialize_mask_info(mask_load_info)


    while True:


        #--------------------phase 0: read the frame and gray scale frame--------------------------------
        # Capture video feed
        ret, frame = video_capture.read()
        # we use frame to detect and segment
        # and use render to render image and imshow
        render = frame.copy()

        print frame.shape

        # Create greyscale image from the video feed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --------------------phase 1: display all the masks at the left side of the frame----------------
        render = exhibit_mask(frame, render, mask_dict, mask_load_info)

        # --------------------phase 2: add mask to face-------------------
        # load face and nose Classifiers
        (faceCascade, noseCascade) = load_Classifiers()

        # add mask to face
        render, face_top, face_bottom, face_left, face_right = add_mask_to_face(render,gray, faceCascade, noseCascade, mask_dict, mask_load_info)

        # ---------------------phase 3: finger detection-------------------------
        render, extreme_top = finger_detect(frame, render, face_top, face_bottom, face_left, face_right)

        # ---------------------phase 4: hook the finger and the mask-------------

        render = drag_mask(frame, render,  extreme_top, mask_dict, mask_load_info)

        # ---------------------phase 5: drag the mask to face--------------------
        # ---------------------phase 6: overlay mask to face---------------------
        # Display the resulting frame

        cv2.imshow('Video', render)

        # press any key to exit
        # NOTE;  x86 systems may need to remove: " 0xFF == ord('q')"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    
