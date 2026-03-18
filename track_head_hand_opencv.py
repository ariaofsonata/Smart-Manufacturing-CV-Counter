import cv2
import numpy as np

# 載入內建的人臉偵測器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. 先偵測臉部位置
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 2. 膚色偵測邏輯
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 3. **核心關鍵**：把臉部的區塊從膚色遮罩中「塗黑」排除
    for (x, y, w, h) in faces:
        # 稍微擴大一點範圍，確保連脖子都排除
        cv2.rectangle(mask, (x, int(y*0.9)), (x+w, y+h+50), (0), -1) 
        # 在原圖畫個框標示臉部（綠色）
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 4. 找剩下的最大區塊（這時候應該就是手了）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        max_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_cnt) > 3000:
            hx, hy, hw, hh = cv2.boundingRect(max_cnt)
            # 畫出手部框（藍色）
            cv2.rectangle(frame, (hx, hy), (hx+hw, hy+hh), (255, 0, 0), 2)
            cv2.putText(frame, "Real Hand", (hx, hy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Face vs Hand Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()