# Smart Manufacturing CV Counter & Motion Tracker

![Python Version](https://img.shields.io/badge/python-3.12-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green)

這個專案展示了如何利用電腦視覺（Computer Vision）技術，在不使用昂貴深度學習模型的情況下，達成高效能的「非接觸式工業零件計數」與「人臉/手部動態追蹤」。

## 🚀 核心功能

* **捏合觸發計數 (Hand Pinch Counter)**: 通過偵測食指與拇指的捏合動作，達成非接觸式的數值累加。
* **人臉排除追蹤 (Face-Filtered Hand Tracking)**: 自動偵測人臉位置並在膚色遮罩中將其排除，有效解決臉部皮膚對手部偵測的干擾。

## 📺 成果展示 (Demo)

| 追蹤頭部與手部動作 | 以手指捏合進行計數 |
| :---: | :---: |
| ![Track Demo](videos/Face_vs_Hand_Detection.gif) | ![Count Demo](videos/Hand_Count.gif) |

## 🛠️ 技術亮點與挑戰克服

### 1. 膚色偵測優化 (HSV Masking)
使用 HSV 色彩空間取代 RGB，能更穩定地捕捉不同光影下的膚色區域。

### 2. 雜訊排除邏輯
* **人臉屏蔽**：利用 `Haar Cascade` 偵測臉部座標，並在 Mask 層將該區域「塗黑」，解決了背景與臉部皮膚導致的誤判問題。
* **輪廓穩定化**：使用 `approxPolyDP` 多邊形逼近法簡化手部邊緣，大幅提升了 `convexityDefects` 運算的穩定性，解決了索引非單調（Not Monotonous）導致的閃退問題。

### 3. 硬體驅動故障排除 (Hardware Troubleshooting)
在開發過程中成功克服了 Windows 10/11 系統下的攝影機驅動衝突（Camera Backend 報錯），並針對特定筆電硬體（Fn 開關設定）進行了底層診斷。

## 📦 安裝與執行

1. **複製專案**:
   ```bash
   git clone https://github.com/ariaofsonata/Smart-Manufacturing-CV-Counter.git
   cd Smart-Manufacturing-CV-Counter

2. **建立虛擬環境與安裝依賴**:

    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    pip install -r requirements.txt

3. **執行程式**:

    計數功能: python hand_count_opencv.py

    追蹤功能: python track_head_hand_opencv.py

👨‍💻 作者
Liao Yuan-shih (ariaofsonata)

11 年機械工程開發經驗 | 數據分析專案經理
專長：Python 自動化、工業視覺原型開發、數據視覺化