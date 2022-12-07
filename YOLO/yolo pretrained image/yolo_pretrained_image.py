#https://github.com/mertfozzy/Live-Stream-Object-Detection/blob/main/stream.py

import cv2
import numpy as np

path = r'D:\Tekonofest\yoloV4-workspace\YOLO\yolo pretrained image\images\people.jpg'

path2 = r'D:\Tekonofest\yoloV4-workspace\YOLO\pretrained_model\yolov3.cfg'

path3 = r'D:\Tekonofest\yoloV4-workspace\YOLO\pretrained_model\yolov3.weights'

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

# kutucuk renkleri ayarlıyoruz :
# buradaki kodları teker teker konsola yazınca değer oluşuyor
colors = ["0,255,255", "0,0,255", "255,0,0", "255,255,0", "0,255,0"]
colors = [np.array(color.split(",")).astype("int") for color in colors]
colors = np.array(colors)
colors = np.tile(colors, (18, 1))

# modeli import ediyoruz : algoritma başlangıcı
model = cv2.dnn.readNetFromDarknet(path2,path3)
layers = model.getLayerNames()

# çıktı katmanlarını araştırıyoruz
output_layer = [layers[layer - 1] for layer in model.getUnconnectedOutLayers()]

model.setInput(img_blob)

# çıktı katmanlarını detectiona sokuyoruz
detection_layers = model.forward(output_layer)
# çıktı katmanlarının içindeki değerleri almış olduk.




























