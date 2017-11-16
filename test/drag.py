import cv2
import imutils
import numpy as np

global test


def cal_dis(x1, y1, x2, y2):
    return np.sqrt( np.power(x1 - x2, 2) + np.power(y1 - y2, 2) )


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

    # print extreme_top

    return extreme_top, extreme_bottom, extreme_left, extreme_right


def segment(frame):
    # # Convert to HSV color space
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #
    #
    # # Create a binary image with where white will be skin colors and rest is black
    # mask2 = cv2.inRange(hsv, np.array([-25, 50, 40]), np.array([25, 153, 255]))

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

def resize_mask(imgDecoration, orig_mask, orig_mask_inv,  DecorationWidth,DecorationHeight):
    resized_imgDecoration = cv2.resize(imgDecoration, (DecorationWidth, DecorationHeight), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(orig_mask, (DecorationWidth, DecorationHeight), interpolation=cv2.INTER_AREA)
    resized_mask_inv = cv2.resize(orig_mask_inv, (DecorationWidth, DecorationHeight), interpolation=cv2.INTER_AREA)

    return (resized_imgDecoration, resized_mask, resized_mask_inv)

def drag_mask(frame, extreme_top,imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth):
    # get the frame height and width
    (frame_height, frame_width) = frame.shape[:2]
    # draw the exhibition area for the masks
    top, bottom, left, right = 0, int(0.3 * frame_height), 0, int(0.2 * frame_width)

    # resize the mask1
    DecorationWidth = right - left
    DecorationHeight = bottom - top

    pic_exh_center = ( (left + right) / 2, (top + bottom) / 2 )

    (resized_imgDecoration, resized_mask, resized_mask_inv) = resize_mask(imgDecoration, orig_mask, orig_mask_inv,
                                                                          DecorationWidth, DecorationHeight)

    # print DecorationHeight, DecorationWidth

    top_mask, bottom_mask, left_mask, right_mask = extreme_top[1] - (DecorationHeight) / 2, extreme_top[1] + (DecorationHeight) / 2 + 1, \
                                                   extreme_top[0] - (DecorationWidth) / 2, extreme_top[0] + (DecorationWidth) / 2

    # print top_mask
    # print bottom_mask
    # print left_mask
    # print right_mask

    if top_mask < 0 or bottom_mask > frame_height or left_mask < 0 or right_mask > frame_width:
        return frame

    dis_to_pic = cal_dis(extreme_top[0], extreme_top[1], pic_exh_center[0], pic_exh_center[1])

    print dis_to_pic

    if dis_to_pic < 30:
        roi_mask = frame[top_mask:bottom_mask, left_mask:right_mask]

        # roi_mask = frame[top:bottom, left:right]

        dst = add_mask_to_ROI(resized_imgDecoration, resized_mask, resized_mask_inv, roi_mask)
        frame[top_mask:bottom_mask, left_mask:right_mask] = dst

    else:
        roi_mask = frame[top:bottom, left:right]

        dst = add_mask_to_ROI(resized_imgDecoration, resized_mask, resized_mask_inv, roi_mask)
        frame[top:bottom, left:right] = dst

    return frame


# main function
if __name__ == "__main__":

    # get the reference to the webcame
    camera = cv2.VideoCapture(0)

    top, right, bottom, left = 10, 350, 225, 590

    # keep looping, until interrupted

    while (True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view ????????????
        frame = cv2.flip(frame, 1)

        # clone the crmae*******
        clone = frame.copy()

        # get the ROI
        roi = frame[top:bottom, right:left]


        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        # roi = frame[top:bottom, left:right]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is read
        # so that our running average models gets calibrated

        # load mask
        (imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth) = load_masks("res/mustache.png" )





        # segment the hand region
        hand = segment(frame)

        # check whether hand region is segmented
        if hand is not None:
            # if yes, up[ack the threshold image and segmented region
            (thresholded, segmented) = hand



            cv2.drawContours(clone, segmented, -1, (0, 0, 255), 3)

            # cv2.imshow("Thresholded", thresholded)

            # count the number of fingers
            extreme_top, extreme_bottom, extreme_left, extreme_right = get_extreme_points(thresholded, segmented)
            #
            #

            clone = drag_mask(clone, extreme_top,imgDecoration, orig_mask, orig_mask_inv, origDecorationHeight, origDecorationWidth)

            #
            #
            cv2.circle(clone, extreme_top, 8, (0, 0, 255), -1)
            # cv2.circle(clone, extreme_bottom, 8, (0, 0, 255), -1)
            # cv2.circle(clone, extreme_left, 8, (0, 0, 255), -1)
            # cv2.circle(clone, extreme_right, 8, (0, 0, 255), -1)

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # break by pressing the "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

