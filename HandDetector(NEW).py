import mediapipe as mp
import cv2

class HandDetector:
    __max_hands = 1
    def __init__(self, mode = False, max_hands = 2, complexity = 1, detectionCon =  0.5, trackingCon = 0.5):
        self.__mpHands = mp.solutions.hands
        self.__mpDraw = mp.solutions.drawing_utils
        self.__myHands = self.__mpHands.Hands(mode, max_hands, complexity, detectionCon, trackingCon)

    def detectHands(self, image, draw = False):
        landmark_lst = []
        image = cv2.flip(image, 1)
        __result = self.__myHands.process(image)
        if __result.multi_hand_landmarks:
            for landmarks in __result.multi_hand_landmarks:
                for id, landmark in enumerate(landmarks.landmark):
                    h, w, c = image.shape
                    x, y = int(w * landmark.x), int(h * landmark.y)
                    landmark_lst.append([id, x, y])
                if draw:
                    self.__mpDraw.draw_landmarks(image, landmarks, self.__mpHands.HAND_CONNECTIONS)
        else:
            landmark_lst = [[None] for i in range(0, 21)]
        return [image, landmark_lst]

    def withinRegionAndHandNavigator(self, image, pt1, pt2, pt3, ROI = [], draw = False):
        __direction = ''
        h, w, c = image.shape
        x, y = w // 2, h // 2
        Default_ROI = [x - 50, y - 50, x + 40, y + 40]
        if pt1[0] != None and pt2[0] != None and pt3[0] != None:
            x1, x2, x3 = pt1[1], pt2[1], pt3[1]
            y1, y2, y3 = pt1[2], pt2[2], pt3[2]
            cx = (x1 + x2 + x3) // 3
            cy = (y1 + y2 + y3) // 3
            if draw:
                if len(ROI) == 0:
                    cv2.rectangle(image, (Default_ROI[0], Default_ROI[1]), (Default_ROI[2], Default_ROI[3]), (0, 0, 255), 5)
                else:
                    cv2.rectangle(image, ROI, (0, 0, 255), 2)
                    cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            if cx > ROI[2]:
                # print('Move left')
                __direction = 'Move left'
            elif cx < ROI[0]:
                # print('Move Right')
                __direction = 'move right'
            elif cy > ROI[3]:
                # print('Move Up')
                __direction = 'Move up'
            elif cy < ROI[1]:
                # print('Move Down')
                __direction = 'Move down'
            if cx in range(ROI[0], ROI[2]) and cy in range(ROI[1], ROI[3]):
                cv2.rectangle(image, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (0, 255, 0), 2)
                # print('Lower hand to pick up')
                __direction = 'Lower hand to pick up'
        return [image, __direction]



