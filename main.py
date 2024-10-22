import cv2
import matplotlib
import numpy

main_image = cv2.imread('main_image.PNG')
image1 = cv2.imread('image1.PNG')
image2 = cv2.imread('image2.PNG')

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

kp1, des1 = sift.detectAndCompute(main_image, None)
kp2, des2 = sift.detectAndCompute(image1, None)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x:x.distance)

output_image = cv2.drawMatches(main_image, kp1, image1, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('SIFT Keypoints', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
