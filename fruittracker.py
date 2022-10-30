#ALL Imports
import cv2
import numpy as np
import torch


#creating Letterbox function
def letterbox(im, new_shape=(1920,2560), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

class FruitTracker():
    def __init__(self,repo='C:\\Users\\hp\\Downloads\\yolov5-master\\yolov5-master',model_path="C:\\Users\\hp\\Desktop\\Telegram Files\\best.pt",device = 0):
        self.model = torch.hub.load(repo, 'custom',
                               (model_path), {'device': device}, source='local')


    def predict(self,img):
        image,r,_ = letterbox(img)
        results = self.model(image)
        startx,starty,endx,endy,pred =[],[],[],[],[]
        results.print()
        for (startX, startY, endX, endY, prob, name) in results.xyxy[0]:
            startx.append(int(startX/r))
            starty.append(int(startY/r))
            endx.append(int(endX/r))
            endy.append(int(endY/r))
            pred.append(name.type(torch.int8))
        return startx,starty,endx,endy,pred

if __name__ =='__main__':
    ft = FruitTracker()
    cap= cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        stx,sty,endx,endy,pred = ft.predict(img)
        if pred!=[]:
            for (startX, startY, endX, endY, name) in zip(stx,sty,endx,endy,pred):
                # draw the bounding box and label on the image
                cv2.rectangle(img,(startX, startY),
                              (endX, endY),
                              (0, 255, 0), 2)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                print(name)
                cv2.putText(img, str(name), (startX, startY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
