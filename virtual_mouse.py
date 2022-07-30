import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

width_cam, height_cam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 17
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, width_cam)
cap.set(4, height_cam)

detector = htm.handDetector(maxHands=1)
width_screen, height_screen = autopy.screen.size()

# print(width_screen, height_screen)

while True:
    # 1. Find the hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lm_list, bbox = detector.findPosition(img)
    # print(lm_list)
    
    # 2. Get the tip of the index and middle fingers
    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]
        # print(x1, y1, x2, y2)
    
        # 3. Check which finger is up
        fingers = detector.fingersUp(img)
        #print(fingers)
        cv2.rectangle(img, (frameR, frameR), 
                       (width_cam-frameR, height_cam-frameR),
                       (255,80,80), 2)
    
        # 4. Only index finger means - moving mode
        # index finger is up and middle finger is down
        if fingers[1] == 1 and fingers[2]==0:
            
            # 5. Comvert coordinates to send the coordinates to mouse

            x3 = np.interp(x1, (frameR, width_cam-frameR), (0, width_screen))
            y3 = np.interp(y1, (frameR, height_cam-frameR), (0, height_screen))
            
            # 6. Smoothing the values to reduce mouse flickr
            clocX = plocX + (x3-plocX)/smoothening
            clocY = plocY + (y3-plocY)/smoothening
            
            # 7. Move mouse
            autopy.mouse.move(width_screen-clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (80,0,255), cv2.FILLED)
            plocX, plocY = clocX, clocY
            
        # 8. Both fingers are up - Clicking mode
        if fingers[1] == 1 and fingers[2]==1:
            # 9. In click mode, we will find the distance between fingers
            length, img, line_info = detector.findDistance(8, 12, img)
            #print(length)
            
            # 10. Click mouse when distance <= threshold
            if length < 45:
                cv2.circle(img, (line_info[4], line_info[5]), 
                           15, (0,255,0), cv2.FILLED)
                autopy.mouse.click()
    
    # 11. Frame Rate
    cTime = time.time() # Current time cTime; previous time pTime
    fps = 1/(cTime-pTime) # Frames per sec
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), 
                cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    
    # 12. Display
    cv2. imshow("Image", img) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()