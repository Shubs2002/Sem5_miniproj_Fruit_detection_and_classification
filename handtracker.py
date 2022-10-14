import cv2
import mediapipe as mp
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

from imutils.object_detection import non_max_suppression

import imutils
WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 224
ROI_SIZE = (224,224)
INPUT_SIZE = (224,224)
W = 640
inprob = 0.85
pos = (0,0)
def sliding_window(image, step, ws):
    # slide a window across the image
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            # yield the current window
            yield (x, y, np.array(image[y:y + ws[1], x:x + ws[0]]))

def image_pyramid(image, scale=1.5, minSize=(224, 224)):
    # yield the original image
    yield image
    # keep looping over the image pyramid
    while True:
        # compute the dimensions of the next image in the pyramid
        # if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
        #     break
        w = 224
        image = imutils.resize(np.array(image), width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
		# yield the next image in the pyramid
        yield image

cap = cv2.VideoCapture(0)
labels_name = ['Lime_Bad','Guava_Bad','Pomegranate_Good','Lime_Good','Apple_Bad','Orange_Bad','Guava_Good','Apple_Good','Orange_Good','Banana_mixed','Banana_Good','Banana_Bad','Pomegranate_Bad','Guava_mixed','Lemon_mixed','Pomegranate_mixed','Apple_mixed','Orange_mixed']

# mpHands = mp.solutions.hands
# hands = mpHands.Hands(static_image_mode=False,
#                       max_num_hands=2,
#                       min_detection_confidence=0.5,
#                       min_tracking_confidence=0.5)
# mpDraw = mp.solutions.drawing_utils
from keras.models import load_model
model = load_model("C:\\Users\\hp\\Desktop\\Indian Fruit_99.69.h5")
# model = VGG16(weights = "C:\\Users\\hp\\Desktop\\Indian Fruit_99.69.h5",include_top = False,input_tensor = Input(shape = (32,640,3)))

pTime = 0
cTime = 0

# while True:
#     success, img = cap.read()
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(imgRGB)
#     #print(results.multi_hand_landmarks)
#     xl1,xr1,xl2,xr2=0,0,999,999
#     yl1,yr1,yl2,yr2=0,0,999,999
#     # print(results.multi_handedness)
#     if results.multi_hand_landmarks:
#         i=0
#         for handLms in results.multi_hand_landmarks:
#
#             for id, lm in enumerate(handLms.landmark):
#
#                 #print(id,lm)
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x *w), int(lm.y*h)
#                 if i==0:
#                     if xl1<cx:
#                         xl1=cx
#                     if xl2>cx:
#                         xl2=cx
#                     if yl1<cy:
#                         yl1=cy
#                     if yl2>cy:
#                         yl2=cy
#                 if i==1:
#                     if xr1 < cx:
#                         xr1 = cx
#                     if xr2 > cx:
#                         xr2 = cx
#                     if yr1 < cy:
#                         yr1 = cy
#                     if yr2 > cy:
#                         yr2 = cy
#                 # if id in range(0,25):
#                 #     #cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
#                 #     x+=cx
#                 #     y+=cy
#             i = i + 1
#             # mpDraw.draw_landmarks(img, handLms)
#             # cv2.circle(img, (x//25, y//25), 3, (255, 0, 255), cv2.FILLED)
#             if xl1!=0 and xl2!=999:
#                 cv2.rectangle(img,(xl1,yl1),(xl2,yl2),(255,0,0),2)
#             if xr1!=0 and xr2!=999:
#                 cv2.rectangle(img,(xr1,yr1),(xr2,yr2),(0,255,0),2)
#
#
#     cTime = time.time()
#     fps = 1/(cTime-pTime)
#     pTime = cTime
#
#     cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
#
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)
scale_x = 14
scale_y = 1
while True:
    success, img = cap.read()
    # shape = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img1 = cv2.resize(imgRGB,(448,640),cv2.INTER_LINEAR)
    # img1 = tf.image.resize(img1 , [448,640])
    img1 = imgRGB
    # img1 = np.expand_dims(img1,axis =0)
    # x,y,z = img1
    # img1 = tf.reshape(imgRGB,(1,224,224,3))
    # imgRGB = cv2.resize(imgRGB,None,interpolation = cv2.INTER_LINEAR ,fx = 2 , fy = 2)
    # results = model.predict(img1)
    # np.array(results)
    # initialize the image pyramid
    # pyramid = image_pyramid(img1, scale=1.5)
    # initialize two lists, one to hold the ROIs generated from the image
    # pyramid and sliding window, and another list used to store the
    # (x, y)-coordinates of where the ROI was in the original image
    # if np.argmax(np.array(model.predict()))
    rois = []
    locs = []
    labels = {}
    # loop over the image pyramid
    # for image in pyramid:
    # determine the scale factor between the *original* image
    # dimensions and the *current* layer of the pyramid
    # scale = W / float(img1.shape[1])
    scale = 1
    # for each layer of the image pyramid, loop over the sliding
    # window locations
    for (x, y, roiOrig) in sliding_window(img1, WIN_STEP, ROI_SIZE):
        # scale the (x, y)-coordinates of the ROI with respect to the
        # *original* image dimensions
        x = int(x * scale)
        y = int(y * scale)
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)
        # take the ROI and preprocess it so we can later classify
        # the region using Keras/TensorFlow
        print(roiOrig.shape)
        # roi = cv2.resize(roiOrig, INPUT_SIZE)
        # roi = img_to_array(roi)
        # roi = preprocess_input(roi)
        # update our list of ROIs and associated coordinates
        rois.append(roiOrig)
        locs.append((x, y, x + w, y + h))
        # check to see if we are visualizing each of the sliding
        # windows in the image pyramid
    preds = []
    for roi in rois:
        roi = np.expand_dims(roi,axis =0)
        # roi = tf.reshape(imgRGB,(1,224,224,3))
        pred = model.predict(roi)
        print(pred)
        pred = np.reshape(pred,18)
        preds.append([np.argmax(pred),pred[np.argmax(pred)]])
        # if np.argmax(preds[-1])<0.85:
        #     preds = preds[:-1]
            # if True:
            #     # clone the original image and then draw a bounding box
            #     # surrounding the current region
            #     clone = img.copy()
            #     cv2.rectangle(clone, (x, y), (x + w, y + h),
            #                   (0, 255, 0), 2)
            #
            #     # show the visualization and current ROI
            #     cv2.imshow("Visualization", clone)
            #     cv2.imshow("ROI", roiOrig)
            #     cv2.waitKey(0)
    # loop over the predictions
    for (i, p) in enumerate(preds):
        # grab the prediction information for the current ROI
        # (imagenetID, label, prob) = p[0]

        # filter out weak detections by ensuring the predicted probability
        # is greater than the minimum probability
        if p[1] > 0.9:
            # grab the bounding box associated with the prediction and
            # convert the coordinates
            box = locs[i]
            label = labels_name[p[0]]
            # grab the list of predictions for the label and add the
            # bounding box and probability to the list
            L = labels.get(label, [])
            L.append((box, p[1]))
            labels[label] = L
    # extract the bounding boxes and associated prediction
    # probabilities, then apply non-maxima suppression
    for label in labels.keys():
        if WIN_STEP > 64:
            WIN_STEP = WIN_STEP // 2
        # loop over all bounding boxes for the current label
        boxes = np.array([p[0] for p in labels[label]])
        proba = np.array([p[1] for p in labels[label]])
        boxes = non_max_suppression(boxes, proba)
        for (startX, startY, endX, endY) in boxes:
            # draw the bounding box and label on the image
            cv2.rectangle(img, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(img1, label , (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # pos[0],pos[1] = startX ,startY
        else :
           if WIN_STEP < 224:
               WIN_STEP*=2
        break

    # show the output after apply non-maxima suppression
    cv2.imshow("Image",img)
    cv2.waitKey(1)

    # loop over all bounding boxes that were kept after applying
    # non-maxima suppression



    #
    # cv2.imshow("Image", img)
    # cv2.waitKey(1)
