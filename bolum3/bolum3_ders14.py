import cv2

path = r'D:\Tekonofest\yoloV4-workspace\image.jpg'

img = cv2.imread(path)
cv2.namedWindow("image",cv2.WINDOW_NORMAL)

cv2.imshow("image",img)

cv2.imwrite(path,img)

cv2.waitKey(0)
cv2.destroyAllWindows()