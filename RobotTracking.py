import numpy as np
import cv2

filepath = "./Match MP4s/F1M2Bensalem2023.mp4" #F1M2 2023 FMA District Bensalem Event

cap = cv2.VideoCapture("C:/Users/lukec/OneDrive/Documents/.Coding/Independent Study/Robotics Scouting/Match MP4s/F1M2Bensalem2023.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

#Object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history = 200, varThreshold=5)
# object_detector = cv2.createBackgroundSubtractorKNN()


while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    #Region of interest
    roi = frame[320:450, 150:1100]
    # cv2.imshow("ROI", roi)

    # hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    

    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array([[(200, 310), (1060, 300), (1200, 400), (1200, 450), (80, 450), (80, 400)]]) 
    cv2.fillPoly(mask, points, (255))

    res = cv2.bitwise_and(frame,frame,mask = mask)

    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]


    gamePiece_roi = cropped

    # mask1 = np.zeros((height, width)) 
    # cv2.fillPoly(mask1, points, (255))
    # # res = cv2.bitwise_and(frame, frame, mask = mask1)
    # rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect 
    # cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    # cv2.fillPoly(mask1, [np.array(gamePiece_roi)], 0)
    # gamePiece_roi = frame

    #Optionally you can remove the crop the image to have a smaller one


    # gamePiece_roi = [(300, 80), (300, 1200), (450, 80), (450, 1200)]  # (x, y)
    # gamePiece_roi = frame[300:450, 80:1200]
    hsv = cv2.cvtColor(gamePiece_roi, cv2.COLOR_BGR2HSV)

### RED ROBOTS
    #Red Mask Values
    lower_red = np.array([150,130,10])
    upper_red = np.array([200,255,150])
    
    
    #Color Mask
    redMask = cv2.inRange(hsv, lower_red, upper_red)
    redRes = cv2.bitwise_and(gamePiece_roi,gamePiece_roi, mask= redMask)
    
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5, 5), np.uint8)
    
    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    redRes = cv2.dilate(redRes, kernel, iterations=3)
    redRes = cv2.erode(redRes, kernel, iterations=2)

    # split hsv to convert to grayscale 
    hR, sR, vR = cv2.split(redRes)

    #take only the value channel to have grayscale
    redMask = vR

    contours, _ = cv2.findContours(redMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        if area > 300:
            cv2.rectangle(gamePiece_roi, (x, y), (x+w, y+h), (0, 0, 255))
            # cv2.drawContours(roi, [cnt], -1, (0, 50, 255), 2)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # cv2.imshow("Red Mask", redMask)

### BLUE ROBOTS

    #Blue Mask Values
    lower_blue = np.array([88,131,0])
    upper_blue = np.array([120,211,252])


    blueMask = cv2.inRange(hsv, lower_blue, upper_blue)
    blueRes = cv2.bitwise_and(gamePiece_roi,gamePiece_roi, mask= blueMask)

    # blueRes = cv2.dilate(blueRes, kernel, iterations=3)
    # blueRes = cv2.erode(blueRes, kernel, iterations=2)
    
    hB, sB, vB = cv2.split(blueRes)
    #Motion Mask
    # redMask = object_detector.apply(redMask, learningRate= -1)
    # blueMask = object_detector.apply(blueMask, learningRate= -1)
    blueMask = 2 * vB
    blueMask = cv2.dilate(blueMask, kernel, iterations=3)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    
    #identify robots in blue mask
    contours, _ = cv2.findContours(blueMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # if bounding box is big enough and a horizontal rectangle or square, draw a tracking box around it
        if 1500 > area > 400 and (8 > w/h > 0.5):
            cv2.rectangle(gamePiece_roi, (x, y), (x+w, y+h), (255, 0, 0))
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # cv2.imshow("BlueMask", blueMask)

#### GAME PIECE DETECTION

    ### CUBES
    lower_purple = np.array([121,135,216])
    upper_purple = np.array([130,255,255])

    purpleMask = cv2.inRange(hsv, lower_purple, upper_purple)
    purpleRes = cv2.bitwise_and(gamePiece_roi, gamePiece_roi, mask= purpleMask)

    hP, sP, vP = cv2.split(purpleRes)

    purpleMask = vP

    # purpleMask = cv2.erode(purpleMask, kernel, iterations=1)
    purpleMask = cv2.dilate(purpleMask, kernel, iterations=1)

    contours, _ = cv2.findContours(purpleMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        if 500 > area > 40:
            cv2.rectangle(gamePiece_roi, (x, y), (x+w, y+h), (255, 255, 0))
            # cv2.drawContours(roi, [cnt], -1, (0, 50, 255), 2)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv2.imshow("PurpleMask", purpleMask)

    ### CONES
    lower_yellow = np.array([9,140,110])
    upper_yellow = np.array([38,255,255])


    yellowMask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellowRes = cv2.bitwise_and(gamePiece_roi, gamePiece_roi, mask= yellowMask)

    # yellowRes = cv2.dilate(yellowRes, kernel, iterations=3)
    # yellowRes = cv2.erode(yellowRes, kernel, iterations=2)
    
    hY, sY, vY = cv2.split(yellowRes)
    #Motion Mask
    # redMask = object_detector.apply(redMask, learningRate= -1)
    # yellowMask = object_detector.apply(yellowMask, learningRate= -1)
    yellowMask = vY
    # yellowMask = cv2.dilate(yellowMask, kernel, iterations=3)
    # yellowMask = cv2.erode(yellowMask, kernel, iterations=2)

    contours, _ = cv2.findContours(yellowMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        if 300 > area > 20:
            cv2.rectangle(gamePiece_roi, (x, y), (x+w, y+h), (0, 255, 255))
            # cv2.drawContours(roi, [cnt], -1, (0, 50, 255), 2)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv2.imshow("YellowMask", yellowMask)

    cv2.imshow("Frame", frame)
    cv2.imshow("Cropped" , gamePiece_roi)

    key = cv2.waitKey(30)
    if key == 27:
        break
  
cap.release()
cv2.destroyAllWindows()