import cv2
from Sem5_miniproj_Fruit_detection_and_classification import fruittracker, HTracker
import Speak

if __name__ =='__main__':
    # ft = FruitTracker(model_path="C:\\Users\\hp\\Desktop\\yolo\\yolov7\\runs\\train\\yolov7\\weights\\epoch_049.pt",repo="C:\\Users\\hp\\Desktop\\yolo\\yolov7")
    ft = fruittracker.FruitTracker()
    hd = HTracker.HandDetector(detectionCon=0.4, max_hands=1)
    sp = Speak.speak
    cap= cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        # img = cv2.flip(img, -1)
        img,stx,sty,endx,endy,pred = ft.predict(img,True)
        img,hlandmarks = hd.detectHands(img, True)
        # print(hlandmarks)
        if pred!=[]:
            img, direction = hd.withinRegionAndHandNavigator(img, hlandmarks[0], hlandmarks[5], hlandmarks[17],
                                                           [int(stx[0]), int(sty[0]), int(endx[0]), int(endy[0])])
            print(pred,direction)
            sp(direction)

        cv2.imshow("Image", img)
        cv2.waitKey(1)