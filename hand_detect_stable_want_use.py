import cv2
import imutils
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




def hsv_method(frame):

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))


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



def ycrcb_method(frame):
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

    # thresh = cv2.threshold(median, 127, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thresh = cv2.threshold(median, 127, 255, cv2.THRESH_BINARY)[1]

    # Find contours of the filtered frame
    _, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return
    else:
        segmented = max(contours, key=cv2.contourArea)
        return (thresh, segmented)

def test_method(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))

    return mask2

# main function
if __name__ == "__main__":


    # get the reference to the webcame
    camera = cv2.VideoCapture(0)


    # keep looping, until interrupted

    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        #flip the frame so that it is not the mirror view ????????????
        frame = cv2.flip(frame,1)

        #clone the crmae*******
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        # roi = frame[top:bottom, left:right]

        #convert the roi to grayscale and blur it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 0)

        # to get the background, keep looking till a threshold is read
        #so that our running average models gets calibrated


        
        # segment the hand region
        hand = hsv_method(clone)



        # check whether hand region is segmented
        if hand is not None:
            # if yes, up[ack the threshold image and segmented region
            (thresholded, segmented) = hand



            cv2.drawContours(clone, segmented, -1, (0, 0, 255), 3)


            #cv2.imshow("Thresholded", thresholded)

            # count the number of fingers
            extreme_top, extreme_bottom, extreme_left, extreme_right = get_extreme_points(thresholded, segmented)
            #
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
# reference
# https://gogul09.github.io/software/hand-gesture-recognition-p1
# https://gogul09.github.io/software/hand-gesture-recognition-p2
