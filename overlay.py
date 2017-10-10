import cv2

# Load two images
img1 = cv2.imread('maxresdefault.jpg')
img2 = cv2.imread('glasses13.png',-1)
# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]

orig_mask = img2[:,:,3]
orig_mask_inv = cv2.bitwise_not(orig_mask)
imgGlass = img2[:,:,0:3]
# Now create a mask of logo and create its inverse mask also
mask = orig_mask
mask_inv = orig_mask_inv
# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(imgGlass,imgGlass,mask = mask)
# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst
cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()