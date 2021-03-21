import cv2
def faceDetect(bgr_image_input):
    gray = cv2.cvtColor(bgr_image_input, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    gray = cv2.adaptiveThreshold(gray, 20, cv2.adaptiveThreshold, cv2.THRESH_BINARY_INV, 5, 0)
    try:
         _, contours, hierarchy = cv2.findContours(gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    except:
         contours, hierarchy = cv2.findContours(gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

i = 0
    contour_id = 0
    #print(len(contours))
    count = 0
    blob_colors = []
    for contour in contours:
        A1 = cv2.contourArea(contour)
        contour_id = contour_id + 1

        if A1 < 3000 and A1 > 1000:
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.01 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            hull = cv2.convexHull(contour)
            if cv2.norm(((perimeter / 4) * (perimeter / 4)) - A1) < 150:
                #if cv2.ma
                count = count + 1
                x, y, w, h = cv2.boundingRect(contour)
                #cv2.rectangle(bgr_image_input, (x, y), (x + w, y + h), (0, 255, 255), 2)
                #cv2.imshow('cutted contour', bgr_image_input[y:y + h, x:x + w])
                val = (50*y) + (10*x)
                blob_color = np.array(cv2.mean(bgr_image_input[y:y+h,x:x+w])).astype(int)
                cv2.drawContours(bgr_image_input,[contour],0,(255, 255, 0),2)
                cv2.drawContours(bgr_image_input, [approx], 0, (255, 255, 0), 2)
                blob_color = np.append(blob_color, val)
                blob_color = np.append(blob_color, x)
                blob_color = np.append(blob_color, y)
                blob_color = np.append(blob_color, w)
                blob_color = np.append(blob_color, h)
                blob_colors.append(blob_color)

