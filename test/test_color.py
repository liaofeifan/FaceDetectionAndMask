import cv2
import imutils
import numpy as np
from skimage.morphology import disk, erosion, dilation, opening, closing, white_tophat,black_tophat, skeletonize
import skimage.filters.rank as sfr


# --------------------------------------------------------------------------
# use hsv to segment the mask and get the finger contour
# i prefer to use HSV when in the classroom
# return (thresh, segmented) or none if not detected
def segment_hybird(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Convert to ycrcb color space
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)


    mask_hsv1 = cv2.inRange(hsv,  np.array([0, 51, 102]), np.array([25 / 2, 153, 255]))
    mask_hsv2 = cv2.inRange(hsv,  np.array([335 / 2, 51, 102]), np.array([360 / 2, 153, 255]))
    mask1 = np.array([])
    mask1 = cv2.bitwise_or(mask_hsv1, mask_hsv2, mask1)

    mask1 = cv2.dilate(mask1, disk(4), iterations=1)

    # mask2 = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    # mask2 = cv2.erode(mask2, disk(4), iterations=1)

    mask = np.array([])
    mask = mask1
    # mask = cv2.bitwise_and(mask1, mask2, mask)
    # mask = cv2.bitwise_and(mask, mask3, mask)
    # cv2.imshow("show", mask)



    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk(5))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk(5))

    erosion = cv2.erode(mask, disk(1), iterations=1)
    dilation = cv2.dilate(erosion, disk(4), iterations=1)
    mask = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, disk(13))


    # eros = erosion(mask, disk(1))
    # dila = dilation(eros, disk(4))

    mask = sfr.median(mask, disk(3))

    # mask = cv2.erode(mask, disk(1), iterations=1)

    cv2.imshow("show", mask)





    # Find contours of the filtered frame
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return
    else:
        segmented = max(contours, key=cv2.contourArea)
    return (mask, segmented)


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




# main function
if __name__ == "__main__":

    # get the reference to the webcame
    camera = cv2.VideoCapture(0)
    camera.set(3, 320)
    camera.set(4, 240)

    # keep looping, until interrupted

    while (True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        # flip the frame so that it is not the mirror view ????????????
        frame = cv2.flip(frame, 1)

        # clone the crmae*******
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        # roi = frame[top:bottom, left:right]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is read
        # so that our running average models gets calibrated



        # segment the hand region
        hand = segment_hybird(clone)

        # check whether hand region is segmented
        #
        #
        # break by pressing the "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


# reference
# https://gogul09.github.io/software/hand-gesture-recognition-p1
# https://gogul09.github.io/software/hand-gesture-recognition-p2





