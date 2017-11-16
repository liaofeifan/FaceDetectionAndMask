import cv2  # OpenCV Library


# Load our overlay image: mustache.png
imgDecoration = cv2.imread('res/ironman.png',-1)

# Create the mask for the mustache
orig_mask = imgDecoration[:,:,3]

# Create the inverted mask for the mustache
# orig_mask_inv = cv2.bitwise_not(orig_mask)
orig_mask_inv = cv2.bitwise_not(orig_mask)
# Convert mustache image to BGR
# and save the original image size (used later when re-sizing the image)
imgDecoration = imgDecoration[:,:,0:3]
origDecorationHeight, origDecorationWidth = imgDecoration.shape[:2]
#-----------------------------------------------------------------------------
#       Main program loop
#-----------------------------------------------------------------------------

# collect video input from first webcam on system
video_capture = cv2.VideoCapture(0)

while True:
    # Capture video feed
    ret, frame = video_capture.read()

    # Create greyscale image from the video feed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # get the frame height and width
    (frame_height, frame_width) = frame.shape[:2]

    # draw the exhibition area for the masks
    top, bottom, left, right = 0, int(0.4 *frame_height), 0, int(0.3 * frame_width)

    # select the mask1 roi
    mask1_roi = frame[top:bottom, left:right]

    #resize the mask1
    DecorationWidth = right - left
    DecorationHeight = bottom - top

    resized_mask = cv2.resize(imgDecoration, (DecorationWidth, DecorationHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(orig_mask, (DecorationWidth, DecorationHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv, (DecorationWidth, DecorationHeight), interpolation=cv2.INTER_AREA)

    # add the mask to the frame
    mask1_roi_bg = cv2.bitwise_and(mask1_roi, mask1_roi, mask=mask_inv)

    mask1_roi_fg = cv2.bitwise_and(resized_mask, resized_mask, mask=mask)

    dst = cv2.add(mask1_roi_bg, mask1_roi_fg)

    frame[top:bottom, left:right] = dst


    # add the rectangle to the frame
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))





    cv2.imshow("exhibition",frame)

    # break by pressing the "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()