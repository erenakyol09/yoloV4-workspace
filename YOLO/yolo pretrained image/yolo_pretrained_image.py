#https://github.com/mertfozzy/Live-Stream-Object-Detection/blob/main/stream.py

import cv2
import numpy as np

path = r'D:\Tekonofest\yoloV4-workspace\YOLO\yolo pretrained image\images\people.jpg'

img = cv2.imread(path)

img_width = img.shape[1]
img_height = img.shape[0]

# bu fonksiyon resmi 4 boyutlu tensora donusturur.
# scale 1/255 girilmistir. Yolo makalesinde bu oneriliyor.
# (416,416) model bu model ile egitildigi icin bu parametre giriliyor.
# swapRB=True resim bgr formatinda buradan rgb formatina dondu
img_blob = cv2.dnn.blobFromImages(img, (1 / 255), (416, 416), swapRB=True, crop=False)

# modelin tanıyacağı labelları giriyoruz :
# önceden indirdiğimiz yolo algoritmasında 80 model var :
labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
          "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
          "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
          "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
          "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
          "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
          "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]

colors = ["0,255,255", "0,0,255", "255,0,0", "255,255,0", "0,255,0"]
colors = [np.array(color.split(",")).astype("int") for color in colors]
colors = np.array(colors)
colors = np.tile(colors, (18, 1))
