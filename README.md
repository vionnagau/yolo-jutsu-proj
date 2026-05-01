# NinjaSight — Real-Time Hand Sign Recognition & AR VFX Pipeline
A computer vision project that detects and classifies Naruto-inspired hand signs in real time using YOLOv8, then validates multi-step jutsu sequences through a custom game state engine. Upon successful recognition, the system triggers interactive AR-style VFX animations rendered with NumPy and OpenCV, synchronized to hand positions tracked by MediaPipe. The solution demonstrates how deep learning and computer vision can be combined to create responsive, low-latency interactive graphics at ~30 FPS.

## Technical Stack

- **Object Detection**: YOLOv8 (trained hand sign classifier)
- **Hand Tracking**: MediaPipe Hand Landmarker (precise 21-point hand pose estimation)
- **Computer Vision**: OpenCV (frame processing, alpha blending, VFX rendering)
- **UI Framework**: Tkinter (cross-platform GUI with real-time video streaming)
- **Deep Learning**: PyTorch (backbone for YOLO inference)

## Project Architecture

### Core Components

1. **detector.py** — YOLOv8-based hand sign classifier
   - Detects and classifies hand signs in real-time
   - Returns bounding boxes with confidence scores
   - Label normalization to match game state

2. **game_state.py** — Game logic and sequence tracking
   - Manages jutsu sequences and player progress
   - Implements hold-duration validation (0.5s minimum per hand sign)
   - Tracks completion state for VFX trigger

3. **vfx_processor.py** — VFX animation rendering engine
   - Loads frame sequences from disk (4 jutsu types)
   - Alpha-blending for smooth overlays
   - Hand position tracking for interactive effects
   - Supports 3 VFX types: hand-follow, full-screen sequence, combo animations

4. **main.py** — Application orchestration
   - Tkinter UI with live video canvas
   - Hand sign icon strip at bottom (progress tracking)
   - Synchronizes detection → game logic → VFX playback
   - MediaPipe integration for effect positioning

### Data Flow

```
Camera Feed
    ↓
[OpenCV] Frame Capture & Preprocessing
    ↓
[YOLOv8] Hand Sign Detection
    ↓
[Game State] Sequence Validation & Hold-Duration Check
    ↓
[Condition: Sequence Complete]
    ├─→ TRUE: Trigger VFX Playback
    │       ↓
    │   [MediaPipe] Hand Position Tracking
    │       ↓
    │   [VFX Processor] Render Frames + Alpha Blend
    │       ↓
    │   [Tkinter] Display on Canvas
    │
    └─→ FALSE: Continue Detection
```

## Features

- **Real-Time Detection**: 30 FPS video processing with YOLOv8
- **Sequential Validation**: Recognize multi-step hand sign sequences (2-6 steps per jutsu)
- **Interactive VFX**: 4 unique jutsu with frame-based animations (28–136 frames each)
- **Hand Tracking**: MediaPipe-based position tracking for effect placement
- **Progress Visualization**: Dynamic UI with icon strip showing current target
- **Smooth Blending**: NumPy vectorized alpha compositing for low-latency VFX

## Supported Jutsus

| Jutsu | Sequence | VFX Type | Frames |
|-------|----------|----------|--------|
| Chidori | Bird → Serpent → Monkey | Hand-Follow | 117 |
| Rasengan | Tiger → Boar | Hand-Follow | 139 |
| Fireball Jutsu | Serpent → Ram → Monkey → Boar → Horse → Tiger | Full-Screen | 34 |
| Shadow Clone Jutsu | Tiger | Combo | 29 |
| Summoning Jutsu | Boar → Dog → Bird → Monkey → Ram | Combo | 29 |

## Installation

### Prerequisites
- Python 3.8+
- Webcam (for real-time detection)
- NVIDIA GPU recommended (for YOLO inference acceleration)

### Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd yolo-jutsu-proj
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify model files:
   - `model/best.pt` — YOLOv8 trained weights
   - `model/hand_landmarker.task` — MediaPipe hand detection model

## Usage

Run from project root:
```bash
python code/main.py
```

**Controls:**
- Show hand signs to camera (clear, confident gestures)
- Hold each sign for 0.5+ seconds
- Complete the sequence to trigger jutsu VFX
- Click **"NEXT JUTSU >>"** to advance to the next jutsu

## Project Structure

```
yolo-jutsu-proj/
├── code/
│   ├── main.py              # Application entry point
│   ├── detector.py          # YOLOv8 inference wrapper
│   ├── game_state.py        # Sequence logic & validation
│   └── vfx_processor.py     # Animation rendering engine
├── model/
│   ├── best.pt              # YOLOv8 weights
│   └── hand_landmarker.task # MediaPipe hand model
├── assets/
│   ├── chidori_frames/      # Chidori animation frames (PNG)
│   ├── rasengan_frames/     # Rasengan animation frames (PNG)
│   ├── fire_frames/         # Fireball animation frames (PNG)
│   ├── smoke_frames/        # Smoke/clone animation frames (PNG)
│   └── *.png                # Hand sign UI icons
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Performance Metrics

- **Detection Latency**: ~33ms per frame (≈30 FPS measured during testing)
- **Confidence Threshold**: set to 0.5 (to reduce false positives during inference)
- **VFX Rendering**: Real-time at 30 FPS (NumPy vectorized pipeline)
- **GPU Memory Usage**: ~1.2 GB (YOLOv8 + MediaPipe on CUDA)

### Training Notes
- Epochs configured: 50  
- Training stopped early at epoch 38 to reduce risk of overfitting  
- Observed trends:  
  - Losses (box, class, DFL) decreased steadily until ~epoch 35, then plateaued  
  - Precision and recall approached ~0.95 by epoch 38  
  - mAP50 reached ~0.97, mAP50-95 stabilized around ~0.70

## Key Achievements

✓ Implemented production-grade real-time computer vision pipeline  
✓ Integrated multiple deep learning models (YOLO + MediaPipe) with minimal latency  
✓ Optimized VFX rendering using NumPy vectorization (no pixel loops)  
✓ Designed robust game state machine with hold-duration validation  
✓ Built intuitive UI with live progress tracking and gesture feedback  

## Technologies & Libraries

- **YOLOv8** — Object detection backbone
- **MediaPipe** — Hand pose estimation
- **OpenCV** — Image processing & alpha blending
- **Tkinter** — GUI framework
- **NumPy** — Numerical optimization
- **Pillow** — Image manipulation

## Future Enhancements

- [ ] GPU-accelerated VFX rendering (CUDA kernels)
- [ ] Multi-hand gesture support
- [ ] Sound effects for jutsu activation
- [ ] Leaderboard system with gesture timing metrics
- [ ] Export VFX sequences as video files
- [ ] Mobile deployment (TensorFlow Lite)

---

**Status**: Production Ready
- Pre-trained Models: Includes the YOLOv8 model and the MediaPipe hand_landmarker.task so you're ready for AI detection out of the box.
- Assets: All the images and VFX files you’ll need to make your project look awesome.
