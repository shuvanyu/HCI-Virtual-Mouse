import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False,
                 maxHands=2,
                 modelComplexity=1,
                 detectionCon=0.5, 
                 trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        # Importing hands module from mediapipe to detect hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,
                                        self.maxHands,
                                        self.modelComplexity,
                                        self.detectionCon,
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
        
    def findHands(self, img, draw=True):
        # Convert BGR (cv2 default format) to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        # process the RGB hand and store the detected hand in results
        self.results = self.hands.process(imgRGB)
        
        # To print and see the hand is detected or not. 
        # Shows Nonne when there is no hand and shows the x, y, z
        # coordinates when the hand is detected
        # print(results.multi_hand_landmarks)
        
        if self.results.multi_hand_landmarks:
            # for loop is needed if there are multiple hands detected
            # by the camera and we want to draw the connections 
            # and outline the hand structure
            for self.handLms in self.results.multi_hand_landmarks:               
                # Drawing the connections of the dots tracing the hand
                if draw:
                    self.mpDraw.draw_landmarks(img, self.handLms, 
                                           self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lm_list = []
        
        if self.results.multi_hand_landmarks:
            #self.results.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(self.handLms.landmark):
                # print(id, lm)
                height, width, channels = img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lm_list.append([id, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx,cy), 5, 
                               (255,0,0), cv2.FILLED)
            
            xMin, xMax = min(xList), max(xList)
            yMin, yMax = min(yList), max(yList)
            
            bbox = xMin, yMin, xMax, yMax
            
            if draw:
                cv2.rectangle(img, (bbox[0]-20, bbox[1]-20), 
                              (bbox[2]+20, bbox[3]+20), (0, 255, 0), 3)
                
        return self.lm_list, bbox
    
    
    def fingersUp(self, img): 
        tipIds = [4, 8, 12, 16, 20]
        if len(self.lm_list) != 0:
            fingers = []
            
            # Thumb            
            if self.lm_list[tipIds[0]][1] > self.lm_list[tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)         
            
            # For fingers      
            for id in range(1, 5):
                if self.lm_list[tipIds[id]][2] < self.lm_list[tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        
        return fingers
    
    
    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lm_list[p1][1], self.lm_list[p1][2]
        x2, y2 = self.lm_list[p2][1], self.lm_list[p2][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        
        # Draw line between the fingers and circles on the finger tips
        if draw:
            cv2.circle(img, (x1,y1), 15, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x2,y2), 15, (0,0,255), cv2.FILLED)
            cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 5)
            cv2.circle(img, (cx,cy), 7, (255,0,0), cv2.FILLED)
                
        length = math.hypot(x2-x1, y2-y1)
        return length, img, [x1, y1, x2, y2, cx, cy]
        

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        # Opening camera to read images/video
        success, img = cap.read()
        img = detector.findHands(img, True)
        lm_list, bbox = detector.findPosition(img)
        #print(lm_list)
        
        if len(lm_list)!= 0:
            fingers = detector.fingersUp(img)
            #print(fingers)
            dist, img, line_info = detector.findDistance(8, 12, img)
            #print(dist)

        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
    
        cv2.putText(img, str(int(fps)), (10, 70), 
                cv2.FONT_HERSHEY_PLAIN, 3, 
                color=(255,0,0), thickness=3)
    
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
