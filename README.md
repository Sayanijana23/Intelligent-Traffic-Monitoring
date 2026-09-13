# 🚦 Intelligent Traffic Monitoring

An AI-powered traffic monitoring system for **vehicle detection, multi-object tracking, vehicle counting, speed estimation, and traffic density analysis** using YOLO, OpenCV, ByteTrack, and Supervision.

---

## ✨ Features

- 🚗 Vehicle Detection using YOLO
- 🎯 Multi-Object Tracking using ByteTrack
- 🔢 Vehicle Counting using line-crossing detection
- 🏎️ Vehicle Speed Estimation using perspective transformation
- 📊 Traffic Density Classification — Low, Medium, High
- 📈 Traffic Analytics
- 💾 Export traffic information to CSV
- 🎥 Generate an annotated output video

---

## 🧠 How It Works

```text
Traffic Video
      ↓
YOLO Vehicle Detection
      ↓
ByteTrack Object Tracking
      ↓
Vehicle Counting
      ↓
Perspective Transformation
      ↓
Speed Estimation
      ↓
Traffic Density Analysis
      ↓
Traffic Analytics
      ↓
Annotated Video + CSV
```

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming |
| YOLO | Vehicle detection |
| Ultralytics | YOLO implementation |
| OpenCV | Video processing and visualization |
| ByteTrack | Multi-object tracking |
| Supervision | Detection, tracking and annotation |
| NumPy | Numerical computations |
| CSV | Traffic analytics storage |

---

## 📂 Project Structure

```text
Intelligent-Traffic-Monitoring/
│
├── data/
│   ├── input_video.mp4
│   └── frame.png
│
├── models/
│   ├── yolov8n.pt
│   └── VisDrone_YOLO_x2.pt
│
├── src/
│   ├── __init__.py
│   ├── annotator.py
│   ├── speed_estimator.py
│   ├── traffic_density.py
│   └── view_transformer.py
│
├── config.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```
## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Sayanijana23/Intelligent-Traffic-Monitoring.git
cd Intelligent-Traffic-Monitoring
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Configure the input video, output path, YOLO model, and perspective transformation points in `config.py`.

## ▶️ Run

```bash
python main.py
```

The system generates:

- Annotated traffic video
- Vehicle count
- Vehicles entering and leaving
- Estimated vehicle speeds
- Traffic density
- Traffic analytics CSV

## 📊 Traffic Analytics

The system records traffic information such as:

| Time     | Vehicle Count | Traffic Density | Vehicles In | Vehicles Out |
|----------|---------------|-----------------|-------------|--------------|
| 10:21:01 | 7             | LOW             | 2           | 1            |
| 10:21:02 | 11            | MEDIUM          | 3           | 2            |
| 10:21:03 | 18            | MEDIUM          | 4           | 3            |

The analytics are saved as `traffic_analysis.csv` for further analysis.

## 📌 Speed Estimation

Perspective transformation is used to map camera-view coordinates to a real-world plane, allowing the system to estimate vehicle speed.

```text
Speed (km/h) = Distance (meters) / Time (seconds) × 3.6
```

## 👩‍💻 Author

**Sayani Jana**  
M.Tech CSE (Data Science), NIT Silchar