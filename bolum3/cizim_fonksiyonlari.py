import  cv2
import numpy as np

def circleRadius():
    return 100

def centerPoint():
    return (256,256)

def redColor():
    return (0,0,255)

def blueColor():
    return (255,0,0)

def greenColor():
    return (0,255,0)

canvas = np.zeros((512,512,3), dtype=np.uint8) + 255

cv2.line(canvas,(256,256),(400,400),(0,255,0),thickness=2)

cv2.circle(canvas,centerPoint(),circleRadius(),blueColor(),thickness=3)


cv2.imshow("Canvas",canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()