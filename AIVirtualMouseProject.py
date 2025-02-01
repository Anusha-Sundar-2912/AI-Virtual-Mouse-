import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import pyautogui  # Added pyautogui for scrolling

#####################
wCam, hCam = 640, 480  # Adjusted height to 480
frameR = 100  # Frame reduction
smoothening = 5
#####################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
plocX, plocY = 0, 0  # Corrected variable name
clocX, clocY = 0, 0  # Corrected variable name

# To get the landmark we need to get the coordinates
detector = htm.handDetector(maxHands=1)  # Only expecting one hand

# To get the width and height of the screen
wScr, hScr = autopy.screen.size()

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)

    # 2. To find the position of the hands
    lmList, bbox = detector.findPosition(img)

    # 3. Get the tip of the index and middle fingers
    if lmList:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 4. Check which fingers are up
        fingers = detector.fingersUp()

        # 5. Draw rectangle for gesture zone
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 6. Only index finger: Moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 7. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 8. Move mouse
            autopy.mouse.move(clocX, clocY)  # Adjusted to use clocX and clocY directly
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 9. Both index and middle finger are up: Clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # Implement clicking logic (distance check, click simulation)
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            # Click mouse if the distance is short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

        # 10. Implement selection feature
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            length, img, lineInfo = detector.findDistance(8, 16, img)
            print(length)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 0, 255), cv2.FILLED)
                autopy.mouse.click()

        # 11. Implement scrolling feature
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            length, img, lineInfo = detector.findDistance ( 8, 12, img )
            print ( length )
            if length > 50:
                pyautogui.scroll ( 5 )  # Scroll up
            elif length < 40:  # Adjusted threshold for scrolling down
                pyautogui.scroll ( -5 )  # Scroll down
            else:
                pass  # No action for intermediate distances

    # 12. Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 13. Display
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
