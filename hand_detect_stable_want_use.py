import cv2
import imutils
import numpy as np

# global variables
bg = None

# To find the running average over the backgound
def run_avg(image, aWeight):
    global bg
    # initialize the backgound
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accmulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)





# hand finger counting
from sklearn.metrics import pairwise
def count(thresholded, segmented):
    # find the convex hull of the segmented hand region
    chull = cv2.convexHull(segmented)
    print "chull = ", len(chull)

    # find the most extreme points in the convex hull
    extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right = tuple(chull[chull[:, :, 0].argmax()][0])

    print "top = ", extreme_top

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



def hsv_method(frame):
    # Blur the image
    blur = cv2.blur(frame, (3, 3))

    # Convert to HSV color space
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

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
    median = cv2.medianBlur(dilation2, 5)

    thresh = cv2.threshold(median, 127, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    # Find contours of the filtered frame
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmented = max(contours, key=cv2.contourArea)

    return (thresh, segmented)


# main function
if __name__ == "__main__":
    # initialize weight
    aWeight = 0.5

    # get the reference to the webcame
    camera = cv2.VideoCapture(0)

    # ROI coordinateds
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0

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
        roi = frame[top:bottom, right:left]
        # roi = frame[top:bottom, left:right]

        #convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 0)

        # to get the background, keep looking till a threshold is read
        #so that our running average models gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = hsv_method(roi)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, up[ack the threshold image and segmented region
                (thresholded, segmented) = hand

                # draw the semgent retion and display the frame****
                # cv2.drawContours(clone, [segmented + (right, top)], -1, (0,0,255))
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                #cv2.imshow("Thresholded", thresholded)

                # count the number of fingers
                fingers = count(thresholded, segmented)

                cv2.putText(clone, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)




        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0))

        # increment the number of frames
        num_frames += 1

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
