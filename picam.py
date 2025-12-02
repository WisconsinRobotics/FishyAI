import cv2 as cv

cap = cv.VideoCapture("mo.mp4")


object_detector = cv.createBackgroundSubtractorMOG2()



while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width, _ = frame.shape
    print(height, width)
    #extract region of interest

    
    blurred = cv.GaussianBlur(frame, (3, 3), 0)
    cv.waitKey(100)
    #Object_detection
    mask = object_detector.apply(blurred)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        
        area = cv.contourArea(cnt)
        if area < 500:
            continue
        
        cv.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
    
    cv.imshow("Frame", frame)
        
    cv.imshow("mask", mask)


cap.release()
cv.destroyAllWindows()

