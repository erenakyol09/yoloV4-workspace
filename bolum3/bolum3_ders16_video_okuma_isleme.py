import cv2

path = r'D:\Tekonofest\yoloV4-workspace\trafik.mp4'
cap = cv2.VideoCapture(path )
#WEBCAM
#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    cv2.imshow("Webcam",frame)
    if cv2.waitKey(30) & 0xFF == ord(" "):
        break

cap.release()
cv2.destroyAllWindows()