# Lost Person Detection

A computer vision project developed as part of a **Data Mining** course to detect faces from videos and identify potential matches against a reference image.

---

## Features

- Face detection from video files or webcam
- Automatic face extraction and cropping
- Multi-stage face verification using:
  - Haar Cascade face detection
  - Eye detection
  - Image preprocessing
  - Confidence scoring
- Duplicate face filtering
- Face similarity matching against a reference image
- Ranked matching results
- Image visualization using Matplotlib

---

## Project Workflow

```
Video Input
     │
     ▼
Image Preprocessing
     │
     ▼
Face Detection
     │
     ▼
Confidence Verification
     │
     ▼
Save Cropped Faces
     │
     ▼
Reference Image
     │
     ▼
Face Matching
     │
     ▼
Ranked Results
```

---

## Project Structure

```
Lost_Person_Detection/
│
├── extract_faces.py
├── facematching.py
├── README.md
├── LICENSE
│
├── haarcascade_frontalface_default.xml
├── haarcascade_profileface.xml
├── haarcascade_eye.xml
│
└── faces/
```

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- Tabulate

---

## Installation

Clone the repository:

```bash
git clone https://github.com/VishwaKrithik/Lost_Person_Detection.git
cd Lost_Person_Detection
```

Install dependencies:

```bash
pip install opencv-python numpy matplotlib tabulate
```

---

## Usage

### 1. Extract Faces

Update the video source in `extract_faces.py` and run:

```bash
python extract_faces.py
```

Detected faces will be saved in the `faces/` directory.

---

### 2. Find Matching Faces

```bash
python facematching.py reference.jpg --dir faces
```

Optional arguments:

```bash
--threshold 0.7
--top 5
--no-display
```

---

## Current Limitations

This project is intended as an academic proof of concept.

- Uses Haar Cascade classifiers for face detection.
- Face matching is based on handcrafted similarity measures rather than deep learning embeddings.
- Performance can be affected by lighting, pose, and image quality.

---

## Future Improvements

- Deep learning face recognition (FaceNet, ArcFace, InsightFace)
- Real-time CCTV support
- Database integration
- Web interface
- GPU acceleration
- Vector search using FAISS

---

## License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

---

## Contributors

- **Vishwa Krithik**
- **Varun Udaykumar**

---

## Disclaimer

This project was developed for **educational and research purposes** as part of a Data Mining course and is not intended for production use.
