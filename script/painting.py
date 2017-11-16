import cv2
import imutils
import numpy as np
from skimage.morphology import disk
import skimage.filters.rank as sfr
from PIL import Image

stop_time = 100

# calculate the distance between (x1, y1) and (x2, y2)
def cal_dis(x1, y1, x2, y2):
    return np.sqrt( np.power(x1 - x2, 2) + np.power(y1 - y2, 2) )

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

    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk(5))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk(5))

    erosion = cv2.erode(mask, disk(1), iterations=1)
    dilation = cv2.dilate(erosion, disk(4), iterations=1)
    mask = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, disk(15))

    # mask = sfr.median(mask, disk(5))


    # Find contours of the filtered frame
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = []
    hull = []
    defects = []
    # if contours is not None:
    #     cnt = contours[0]
    #
    #     hull = cv2.convexHull(cnt, returnPoints=False)
    #     defects = cv2.convexityDefects(cnt, hull)


    if len(contours) == 0:
        return
    else:
        segmented = max(contours, key=cv2.contourArea)
    return (mask, segmented, cnt, hull, defects)




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
    neck_height = 50
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
    hand = segment_hybird(frame)

    extreme_top = [0,0]

    segmented = []

    # check whether hand region is segmented
    if hand is not None:
        # if yes, up[ack the threshold image and segmented region
        (thresholded, segmented, cnt, hull, defects) = hand

        # draw the semgent retion nd display the frame****
        cv2.drawContours(render, segmented, -1, (0, 255, 255), 2)
        # cv2.imshow("Thresholded", thresholded)

        extreme_top, extreme_bottom, extreme_left, extreme_right = get_extreme_points(segmented)

        cv2.circle(render, extreme_top, 8, (0, 0, 255), -1)

        # if defects is not None:
        #     for i in range(defects.shape[0]):
        #         s, e, f, d = defects[i, 0]
        #         start = tuple(cnt[s][0])
        #         end = tuple(cnt[e][0])
        #         far = tuple(cnt[f][0])
        #         cv2.line(render, start, end, [0, 255, 0], 2)
        #         cv2.circle(render, far, 5, [0, 0, 255], -1)

    return render, extreme_top, segmented

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



def painting(frame, render, pts_list, paper_coor, roi_white_paper):
    cv2.rectangle(render,(paper_coor[2], paper_coor[0]), (paper_coor[1], paper_coor[3]), (0,0,155), 1)
    roi_paper = render[paper_coor[0]:paper_coor[3], paper_coor[2]:paper_coor[1]]
    if len(pts_list) != 0:
        for pts in pts_list:
            cv2.polylines(roi_paper, [np.array(pts)], False, (255, 0, 0), 4)
            cv2.polylines(roi_white_paper, [np.array(pts)], False, (255, 0, 0), 4)


    return render, roi_white_paper






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

def is_click(frame, render, extreme_top):
    frame_height, frame_width = frame.shape[:2]

    click_coor = [frame_height - 50, frame_height, 0, 50]

    cv2.rectangle(render, (click_coor[2], click_coor[0]), (click_coor[3], click_coor[1]), (0, 255, 255), 2)

    if extreme_top is not None:
        if (extreme_top[0] <= click_coor[3] and extreme_top[0] >= click_coor[2]) and (extreme_top[1] >= click_coor[0] and extreme_top[1] <= click_coor[1]):
            return True

    return False

def create_alpha_mask(painted_mask):
    h,w = painted_mask.shape[:2]
    to_save_mask = np.zeros((h,w,4), np.uint8)
    for i in range(h):
        for j in range(w):
            to_save_mask[i,j,0], to_save_mask[i,j,1], to_save_mask[i,j,2] = \
                painted_mask[i,j,0], painted_mask[i,j,1], painted_mask[i,j,2]
            to_save_mask[i,j,3] = 155

    return to_save_mask

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


    finger_print_list = [[]]
    stop_count = 0
    w,h = video_capture.get(3), video_capture.get(4)
    paper_coor = [0, int(0.5 * h), 0, int(0.5 * w)]
    painted_mask = np.zeros((paper_coor[1] - paper_coor[0], paper_coor[3] - paper_coor[2], 3), np.uint8)

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

        (render, extreme_top, segmented) = detect_finger(frame, render, face_coor)


        draw_mask(frame, render, segmented, extreme_top, finger_print_list, paper_coor, painted_mask)
        # print painted_mask
        # if is_fist(frame, segmented) is True:
        #
        #
        #
        #
        #     finger_print.append(extreme_top)
        #
        #
        #
        # painting(frame, render, finger_print_list)
        if is_click(frame, render, extreme_top):
            p_h, p_w = painted_mask[0], painted_mask[1]
            # save_painted_mask = np.dstack((painted_mask, np.zeros((painted_mask.shape[:2]))))
            # cv2.imwrite("res/test.png", save_painted_mask)
            # save_painted_mask = cv2.cvtColor(painted_mask, cv2.COLOR_RGB2RGBA)
            save_painted_mask = create_alpha_mask(painted_mask)
            # print save_painted_mask.shape
            cv2.imwrite("res/test.png", save_painted_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        # imshow the frame
        cv2.imshow('Video', render)

        # finger_print = []
        # press any key to exit
        # NOTE;  x86 systems may need to remove: " 0xFF == ord('q')"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()