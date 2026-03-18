import cv2
import numpy as np

# 1. 初始化攝影機與人臉偵測器 (用來排除臉部干擾)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

count = 0
is_touching = False

print("🚀 穩定版計數器啟動...")
print("提示：請將手掌面向鏡頭，捏合食指與拇指進行計數。按 'q' 退出。")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 鏡像翻轉畫面，操作起來比較直覺
    frame = cv2.flip(frame, 1)
    
    # 2. 膚色偵測與人臉排除
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 塗黑臉部區域，避免被誤認為手
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, int(y*0.5)), (x+w, y+h+50), (0), -1)

    # 3. 尋找手部輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找面積最大的輪廓（通常是手）
        max_cnt = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(max_cnt) > 3000:
            # --- 穩定化修正：簡化輪廓以減少雜訊 ---
            epsilon = 0.01 * cv2.arcLength(max_cnt, True)
            approx_cnt = cv2.approxPolyDP(max_cnt, epsilon, True)
            
            # 取得外框
            hx, hy, hw, hh = cv2.boundingRect(approx_cnt)
            
            # 計算凸包索引 (一定要 returnPoints=False)
            hull_indices = cv2.convexHull(approx_cnt, returnPoints=False)
            
            # 4. 偵測捏合邏輯 (利用 Convexity Defects)
            try:
                defects = cv2.convexityDefects(approx_cnt, hull_indices)
                gap_count = 0
                
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        # 距離閥值 d：若間隙深度夠大，代表手指是張開的
                        if d > 15000: 
                            gap_count += 1
                
                # 捏合判斷：當手指間隙消失 (gap_count 變少)
                if gap_count < 2: 
                    if not is_touching:
                        count += 1
                        is_touching = True
                        print(f"✅ 偵測到捏合！目前總計: {count}")
                else:
                    is_touching = False

            except cv2.error:
                # 若發生索引非單調錯誤，略過此幀以防閃退
                pass

            # 5. 畫面視覺化
            color = (0, 0, 255) if is_touching else (0, 255, 0)
            cv2.rectangle(frame, (hx, hy), (hx+hw, hy+hh), color, 2)
            state_text = "TOUCHING!" if is_touching else "Open Hand"
            cv2.putText(frame, state_text, (hx, hy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 顯示 UI 資訊
    cv2.rectangle(frame, (5, 5), (280, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"COUNT: {count}", (20, 55), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 2)
    
    cv2.imshow('Smart Inventory Prototype', frame)
    # cv2.imshow('Mask (Debug)', mask) # 若想看偵測情況可取消註解
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()