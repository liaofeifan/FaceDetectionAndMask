import cv2
import numpy as np
import imutils

img = cv2.imread("res/messi.jpg")

h, w = img.shape[:2]

# resize the image
res = cv2.resize(img, (2 * w, 2 * h), interpolation= cv2.INTER_CUBIC)

# rotate the image
rot = imutils.rotate_bound(img, 30)

print img
print rot

cv2.imshow("show", rot)
cv2.waitKey(0)
cv2.destroyAllWindows()