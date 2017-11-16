import cv2
import imutils
import numpy as np
from skimage.morphology import disk
import skimage.filters.rank as sfr


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


#--------------------------------------------------------------------------
# load (imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth)
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
# load (imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth) in to
# mask_info
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
# reload (imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth) in to
# mask_info
def reload_mask_info(mask_load_info):
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
# calculate the mask coordinate used to be exhibition according to the frame height and width
# and load them in to mask_coordinate
def initialize_mask_dict(mask_coordinate, frame_height, frame_width):
    count = 0
    for mask_name, mask_status in mask_coordinate.iteritems():
        # draw the exhibition area for the masks
        top, bottom, left, right = int(0.25 * frame_height) * count, int((0.25 + count * 0.25) * frame_height), 0, int(
            0.17 * frame_width)

        mask_status[0] = top
        mask_status[1] = bottom
        mask_status[2] = left
        mask_status[3] = right

        count += 1

# --------------------------------------------------------------------------
# initialize the coordinate of masks
# calculate the mask coordinate used to be exhibition according to the frame height and width
# and load them in to mask_coordinate
def reload_mask_dict(mask_coordinate, frame_height, frame_width):
    count = 0
    for mask_name, mask_status in mask_coordinate.iteritems():
        # draw the exhibition area for the masks
        top, bottom, left, right = int(0.25 * frame_height) * count, int(
            (0.25 + count * 0.25) * frame_height), 0, int(
            0.17 * frame_width)

        mask_status[0] = top
        mask_status[1] = bottom
        mask_status[2] = left
        mask_status[3] = right

        count += 1

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

#--------------------------------------------------------------------------
# get mask name by mask_state
def get_mask_name(mask_status, mask_state):
    if mask_status is not None:
        return mask_status.keys()[mask_status.values().index(mask_state)]
    return " "

#--------------------------------------------------------------------------
# calculate the distance between (x1, y1) and (x2, y2)
def cal_dis(x1, y1, x2, y2):
    return np.sqrt( np.power(x1 - x2, 2) + np.power(y1 - y2, 2) )


# --------------------------------------------------------------------------
# use hsv to segment the mask and get the finger contour
# i prefer to use HSV when in the classroom
# return (thresh, segmented) or none if not detected
def segment_hybird(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Convert to ycrcb color space
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

    mask_hsv1 = cv2.inRange(hsv, np.array([0, 51, 102]), np.array([25 / 2, 153, 255]))
    mask_hsv2 = cv2.inRange(hsv, np.array([335 / 2, 51, 102]), np.array([360 / 2, 153, 255]))
    mask1 = np.array([])
    mask1 = cv2.bitwise_or(mask_hsv1, mask_hsv2, mask1)

    mask2 = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))

    mask = np.array([])
    mask = cv2.bitwise_and(mask1, mask2, mask)


    erosion = cv2.erode(mask, disk(1), iterations=1)
    dilation = cv2.dilate(erosion, disk(4), iterations=1)
    mask = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, disk(15))

    mask = sfr.median(mask, disk(5))


    # Find contours of the filtered frame
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return
    else:
        segmented = max(contours, key=cv2.contourArea)
    return (mask, segmented)

#-----------------------------------------------------------------------------
# resize the mask to DecorationWidth and DecorationHeight
# return (resized_img, resized_mask, resized_mask_inv)
def resize_mask(img, orig_mask, orig_mask_inv,  DecorationWidth,DecorationHeight):
    resized_img = cv2.resize(img,  (DecorationWidth, DecorationHeight), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(orig_mask, (DecorationWidth, DecorationHeight), interpolation=cv2.INTER_AREA)
    resized_mask_inv = cv2.resize(orig_mask_inv, (DecorationWidth, DecorationHeight), interpolation=cv2.INTER_AREA)

    return (resized_img, resized_mask, resized_mask_inv)



#-----------------------------------------------------------------------------
# display the mask on render according to the coordinate
# return render
def add_img_to_render(render, img, mask, mask_inv, coor):
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
    top, bottom, left, right = coor[0], coor[1], coor[2], coor[3]

    roi = render[top:bottom, left:right]

    # roi_bg contains the original image only
    #  where the mustache is not
    # in the region that is the size of the mustache.
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # roi_fg contains the image of the mustache only where the mustache is
    roi_fg = cv2.bitwise_and(img, img, mask=mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg, roi_fg)

    render[top:bottom, left:right] = dst

    return render



#-----------------------------------------------------------------------------
# resize the mask according to the coordinate,
# display the mask on render according to the coordinate
# resize according to the frame size
# return render
def resize_and_add_img_to_render(render, mask_name, mask_info, coor):
    top, bottom, left, right = coor[0], coor[1], coor[2], coor[3]
    (frame_height, frame_width) = render.shape[:2]

    if top < 0:
        top = 0
    if bottom > frame_height:
        bottom = frame_height
    if left < 0:
        left = 0
    if right > frame_width:
        right = frame_width

    DecorationWidth = right - left
    DecorationHeight = bottom - top


    if DecorationWidth == 0 and DecorationHeight == 0:
        return render

    (resized_img, resized_mask, resized_mask_inv) = resize_mask(mask_info[mask_name][0],
                                                                mask_info[mask_name][1],
                                                                mask_info[mask_name][2],
                                                                DecorationWidth,
                                                                DecorationHeight)

    to_add_coor = [top, bottom, left, right]

    add_img_to_render(render, resized_img, resized_mask, resized_mask_inv, to_add_coor)

    return render

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
    # cv2.rectangle(render, (face_left, face_top), (face_right, face_bottom), (0, 0, 255), 2)

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
    extra_bottom = 50
    bottom += extra_bottom

    extra_left = 10
    left -= extra_left

    extra_right = 10
    right += extra_right

    extra_top = 10
    top -= extra_top

    if (bottom + 2) > frame_height:
        bottom = frame_height - 3
    if (top - 2) < 0:
        top = 0 + 3
    if (right + 2) > frame_width:
        right = frame_width - 3
    if (left - 2) < 0:
        left = 0 + 3


    cv2.rectangle(render, (left, top), (right, bottom), (0, 0, 255), 2)

    width = right - left
    height = bottom - top

    if width is not 0 and height is not 0:

        frame_roi = frame[top:bottom, left:right]
        black_box = np.zeros(frame_roi.shape, np.uint8)
        # white_mask = np.zeros((height + 2, width + 2), np.uint8)
        # cv2.floodFill(frame_roi, white_mask, (0, 0), (255, 255, 255))
        #
        # frame_roi = cv2.bitwise_not(frame_roi)

        # frame[top:bottom, left:right] = frame_roi
        frame[top:bottom, left:right] = black_box

    # to get the background, keep looking till a threshold is read
    # so that our running average models gets calibrated

    # segment the hand region
    hand = segment_hybird(frame)

    extreme_top = [0,0]

    # check whether hand region is segmented
    if hand is not None:
        # if yes, up[ack the threshold image and segmented region
        (thresholded, segmented) = hand

        # draw the semgent retion nd display the frame****
        cv2.drawContours(render, segmented, -1, (0, 255, 255), 2)
        # cv2.imshow("Thresholded", thresholded)

        extreme_top, extreme_bottom, extreme_left, extreme_right = get_extreme_points(segmented)

        cv2.circle(render, extreme_top, 8, (0, 0, 255), -1)

    return render, extreme_top, segmented


#-----------------------------------------------------------------------------
# exhibit tha masks
def exhibit_masks(frame, render, mask_status, mask_coors, mask_info):
    # exhibit masks which state == 1
    for mask_name, mask_state in mask_status.iteritems():
        if mask_state == 1:
            # this mask need to be exhibited
            resize_and_add_img_to_render(render, mask_name, mask_info, mask_coors[mask_name])


#-----------------------------------------------------------------------------
# test if click on "return" bottom
# return true or false
def is_click(frame, render, extreme_top, region):
    click_coor = region

    cv2.rectangle(render, (click_coor[2], click_coor[0]), (click_coor[3], click_coor[1]), (0, 255, 255), 2)

    if extreme_top is not None:
        if (extreme_top[0] <= click_coor[3] and extreme_top[0] >= click_coor[2]) and (extreme_top[1] >= click_coor[0] and extreme_top[1] <= click_coor[1]):
            return True

    return False


#-----------------------------------------------------------------------------
# set all the mask_states
def set_mask_status(mask_status, state):
    for mask_name in mask_status.keys():
        mask_status[mask_name] = state



#-----------------------------------------------------------------------------
# drag the mask
# return render
def drag(frame, render, extreme_top, face_coor, mask_status, mask_coors, mask_info, state_4_count):
    CONST_STATE_4_SHADOW = 15
    print "state_4_count", state_4_count

    # if a mask is been draged from face to exhibition
    if 4 in mask_status.values():
        state_4_count = 0
        dragged_back_mask_name = get_mask_name(mask_status, 4)
        mask_cen = [(mask_coors[dragged_back_mask_name][2] + mask_coors[dragged_back_mask_name][3]) / 2,
                    (mask_coors[dragged_back_mask_name][0] + mask_coors[dragged_back_mask_name][1]) / 2]
        finger_to_mask = cal_dis(extreme_top[0], extreme_top[1], mask_cen[0], mask_cen[1])

        if finger_to_mask < 30:
            mask_status[dragged_back_mask_name] = 1
            # just find one mask is ok

    # if a mask is beening dragged
    elif 2 in mask_status.values():
        # get the dragging mask name
        dragged_mask_name = get_mask_name(mask_status, 2)
        # calculate the distance between finger and face
        face_cen = [ (face_coor[2] + face_coor[3]) / 2, (face_coor[0] + face_coor[1]) / 2 ]
        r = cal_dis(face_cen[0], face_cen[1], face_coor[2], face_coor[0])
        finger_to_face = cal_dis(extreme_top[0], extreme_top[1], face_cen[0], face_cen[1])
        if abs(r - finger_to_face) < 10:
            # now the finger near the face
            print "near face"
            # if a mask is already been overlayed on face
            if 3 in mask_status.values():
                now_overlay_mask_name = get_mask_name(mask_status, 3)
                # if drag same mask
                if dragged_mask_name == now_overlay_mask_name:
                    mask_status[now_overlay_mask_name] = 3
                    return render, state_4_count
                else:
                    # drag a different mask
                    mask_status[now_overlay_mask_name] = 1
                    mask_status[dragged_mask_name] = 3
            # face is not been overlapped, just add mask to face
            mask_status[dragged_mask_name] = 3
    else:
        # want to drag mask back from face to exhibition
        # calculate the distance between finger and face
        face_cen = [(face_coor[2] + face_coor[3]) / 2, (face_coor[0] + face_coor[1]) / 2]
        r = cal_dis(face_cen[0], face_cen[1], face_coor[2], face_coor[0])
        finger_to_face = cal_dis(extreme_top[0], extreme_top[1], face_cen[0], face_cen[1])
        if abs(r - finger_to_face) < 5:
            if state_4_count < CONST_STATE_4_SHADOW:
                state_4_count += 1
                print "state_4_count < CONST_STATE_4_SHADOW:", state_4_count
                return render, state_4_count
            else:
                if 3 in mask_status.values():
                    state_4_count = 0
                    now_overlay_mask_name = get_mask_name(mask_status, 3)
                    mask_status[now_overlay_mask_name] = 4

        # find finger near which face
        for mask_name, mask_state in mask_status.iteritems():
            if mask_state == 1:
                mask_cen =[ (mask_coors[mask_name][2] + mask_coors[mask_name][3]) / 2, (mask_coors[mask_name][0] + mask_coors[mask_name][1]) / 2]
                finger_to_mask = cal_dis(extreme_top[0], extreme_top[1], mask_cen[0], mask_cen[1])

                if finger_to_mask < 10:
                    mask_status[mask_name] = 2
                    # just find one mask is ok
                    break

    return render, state_4_count


#-----------------------------------------------------------------------------
# display mask according to mask state
def display_mask(frame, render, extreme_top, face_coor, nose_coor, mask_status, mask_coors, mask_info):
    if 5 in mask_status.values():
        return render

    exhibit_masks(frame, render, mask_status, mask_coors, mask_info)

    if 3 in mask_status.values():
        # add mask on face
        overlay_mask_name = get_mask_name(mask_status, 3)

        # need to be fixed


        resize_and_add_img_to_render(render, overlay_mask_name, mask_info, face_coor)

    if 2 in mask_status.values():
        drag_mask_name = get_mask_name(mask_status, 2)
        # print drag_mask_name
        mask_height, mask_width = mask_coors[drag_mask_name][1] - mask_coors[drag_mask_name][0], \
                                  mask_coors[drag_mask_name][3] - mask_coors[drag_mask_name][2]


        drag_coor = [ extreme_top[1] - mask_height / 2, extreme_top[1] + mask_height / 2,
                      extreme_top[0] - mask_width / 2, extreme_top[0] + mask_width / 2 ]



        resize_and_add_img_to_render(render, drag_mask_name, mask_info, drag_coor)

    if 4 in mask_status.values():
        drag_mask_name = get_mask_name(mask_status, 4)
        # print drag_mask_name
        mask_height, mask_width = mask_coors[drag_mask_name][1] - mask_coors[drag_mask_name][0], \
                                  mask_coors[drag_mask_name][3] - mask_coors[drag_mask_name][2]


        drag_coor = [ extreme_top[1] - mask_height / 2, extreme_top[1] + mask_height / 2,
                      extreme_top[0] - mask_width / 2, extreme_top[0] + mask_width / 2 ]

        resize_and_add_img_to_render(render, drag_mask_name, mask_info, drag_coor)

    return render



#-----------------------------------------------------------------------------
# draw the self-mask
def draw_mask(frame, render, segmented, extreme_top, finger_print_list, paper_coor, painted_mask):

    # is one finger
    if is_fist(frame, segmented) is not True:
        finger_print_list[-1].append(extreme_top)

    # is fist
    else:
        if finger_print_list[-1] != []:
            finger_print_list.append([])

    painting(frame, render, finger_print_list, paper_coor, painted_mask)
    # print len(finger_print_list)

    return render, painted_mask


#-----------------------------------------------------------------------------
# helper function for draw_mask
# use to runder mask on render frame and paper
def painting(frame, render, pts_list, paper_coor, roi_white_paper):
    cv2.rectangle(render,(paper_coor[2], paper_coor[0]), (paper_coor[1], paper_coor[3]), (0,0,155), 1)
    roi_paper = render[paper_coor[0]:paper_coor[3], paper_coor[2]:paper_coor[1]]
    if len(pts_list) != 0:
        for pts in pts_list:
            cv2.polylines(roi_paper, [np.array(pts)], False, (255,20, 147), 8)
            cv2.polylines(roi_white_paper, [np.array(pts)], False, (255, 20, 147), 8)


    return render, roi_white_paper


#-----------------------------------------------------------------------------
# detect if the contour detected is fist
def is_fist(frame, segmented):
    if len(segmented) == 0:
        return False
    extreme_top, extreme_bottom, extreme_left, extreme_right = get_extreme_points(segmented)
    cnt = segmented
    whole_area =  cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    moments = cv2.moments(cnt)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
        cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00

    centr = (cx, cy)
    radius_left = cal_dis(centr[0], centr[1], extreme_left[0], extreme_left[1])
    radius_right = cal_dis(centr[0], centr[1], extreme_right[0], extreme_right[1])
    radius = 0
    if radius_left > radius_right:
        radius = radius_right
    else:
        radius = radius_left
    dis = cal_dis(centr[0], centr[1], extreme_top[0], extreme_top[1])
    rat = dis / radius

    if rat > 1.6:
        return False
    return True


#-----------------------------------------------------------------------------
# convert from RGB to RGBA to save
def create_alpha_mask(painted_mask):
    h,w = painted_mask.shape[:2]
    to_save_mask = np.zeros((h,w,4), np.uint8)
    for i in range(h):
        for j in range(w):
            to_save_mask[i,j,0], to_save_mask[i,j,1], to_save_mask[i,j,2] = \
                painted_mask[i,j,0], painted_mask[i,j,1], painted_mask[i,j,2]
            to_save_mask[i,j,3] = 200

    return to_save_mask


if __name__ == "__main__":

    #-----------------------------------------------------------------------------
    # init
    #-----------------------------------------------------------------------------
    video_capture = cv2.VideoCapture(0)

    # set video resolution
    video_capture.set(3,640)
    video_capture.set(4,480)

    # get frame weight and frame height
    w, h = video_capture.get(3), video_capture.get(4)

    # work status
    # status 1: display-drag
    # status 2: draw
    work_state = 1

    # mask_status {"mask_name": mask_state}
    # status 1 : display
    # status 2 : drag from exhibit to face
    # status 3 : add on face
    # status 4 : drag from face to exhibit
    # status 5 : clean the desktop when drawing

    mask_status = {"res/self_mask.png" : 1,  "res/cat_ears2.png" : 1, "res/funny_glasses.png" : 1}

    # mask_coors = {"mask_name": [top, bottom, left, right]}
    mask_coors = {"res/self_mask.png": [0, 0, 0, 0], "res/cat_ears2.png": [0, 0, 0, 0], "res/funny_glasses.png" : [0, 0, 0, 0]}

    #  mask_info = {"mask_name": [imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth]}
    mask_info = {"res/self_mask.png": [[], [], [], [], []],  "res/cat_ears2.png": [[], [], [], [], []], "res/funny_glasses.png" :[[], [], [], [], []]}

    # initialize the coordinate of masks
    initialize_mask_dict(mask_coors, video_capture.get(4), video_capture.get(3))

    # initialize the information of masks
    initialize_mask_info(mask_info)

    # load face, fist and nose Classifiers
    (faceCascade, noseCascade, fistCascade, eyeCascade) = load_Classifiers()


    # count how many frames drag be shadowed
    state_4_count = 0

    # multiple finger tips
    finger_print_list = [[]]

    # roi to self-draw the mask
    paper_coor = [0, int(0.5 * h), 0, int(0.5 * w)]

    # roi to save the mask
    painted_mask = np.zeros((paper_coor[1] - paper_coor[0], paper_coor[3] - paper_coor[2], 3), np.uint8)

    # region of clear button
    clear_button = [int(0.87 * h), int(h), 0, int(0.1 * w)]

    # region of start-draw button
    start_draw_button = [int(0.87 * h), int(h), int(0.13 * w), int(0.23 * w)]

    # region of return-draw button
    return_button = [int(0.87 * h), int(h), int(0.26 * w), int(0.36 * w)]



    while True:

        # print mask_status

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

        (render, extreme_top, segmented) = detect_finger(frame, render, face_coor)

        # -----------------------------------------------------------------------------
        # phase 4: finger point to botton, and change the work statues accordingly
        # -----------------------------------------------------------------------------
        # if click on clear button, set all mask status to 1
        if is_click(frame,render,extreme_top,clear_button):
            set_mask_status(mask_status, 1)

        # if click on draw mask button
        if is_click(frame,render,extreme_top,start_draw_button):
            # let's start drawing mask!
            # change the work state to 2
            work_state = 2
            # and clean all the cache
            finger_print_list = [[]]
            p_h, p_w = painted_mask.shape[0], painted_mask.shape[1]
            white_paper = np.zeros((p_h, p_w, 3), np.uint8)
            save_painted_mask = create_alpha_mask(white_paper)
            cv2.imwrite("res/self_mask.png", save_painted_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        if is_click(frame, render, extreme_top, return_button):
            set_mask_status(mask_status, 1)
            work_state = 1
            p_h, p_w = painted_mask.shape[0], painted_mask.shape[1]
            # and save the mask
            save_painted_mask = create_alpha_mask(painted_mask)
            cv2.imwrite("res/self_mask.png", save_painted_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            # reload the masks
            reload_mask_dict(mask_coors, frame.shape[0], frame.shape[1])
            reload_mask_info(mask_info)



        if work_state == 1:
            # -----------------------------------------------------------------------------
            # phase 5: drag masks
            # -----------------------------------------------------------------------------

            render, state_4_count = drag(frame,render,extreme_top,face_coor,mask_status,mask_coors,mask_info, state_4_count)


        elif work_state == 2:

            # -----------------------------------------------------------------------------
            # phase 7: draw the self-mask
            # -----------------------------------------------------------------------------
            set_mask_status(mask_status, 5)
            draw_mask(frame, render, segmented, extreme_top, finger_print_list, paper_coor, painted_mask)

        # -----------------------------------------------------------------------------
        # phase 6: display mask according to mask state
        # -----------------------------------------------------------------------------

        display_mask(frame, render, extreme_top, face_coor, nose_coor, mask_status, mask_coors, mask_info)

        # imshow the frame
        cv2.imshow('Video', render)

        # press any key to exit
        # NOTE;  x86 systems may need to remove: " 0xFF == ord('q')"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

            # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()