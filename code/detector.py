import os
from ultralytics import YOLO


class JutsuDetector:
    """
    wraps the yolov8 model.
    takes a raw opencv frame and returns a list of hand sign detections.
    """

    def __init__(self, model_path=None, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'best.pt')
        print(f"[DEBUG] Loading YOLO model from: {model_path}")
        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found: {model_path}")
        self.model = YOLO(model_path)
        print(f"[DEBUG] YOLO model loaded successfully")

    def detect(self, frame):
        """
        run inference on one frame.
        returns: list of (label, confidence, (x1, y1, x2, y2))
        """
        detections = []
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        for r in results:
            for box in r.boxes:
                label = self.model.names[int(box.cls[0])]
                label = label.strip().strip('-').lower()  # Remove hyphens and whitespace
                conf  = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((label, conf, (x1, y1, x2, y2)))
                print(f"[DETECTOR] Found: {label} (conf: {conf:.2f})")

        if not detections:
            print(f"[DETECTOR] No detections in this frame (conf_threshold: {self.conf_threshold})")
        
        return detections