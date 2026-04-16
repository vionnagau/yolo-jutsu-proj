from ultralytics import YOLO

class JutsuDetector:
    def __init__(self, model_path="../model/best.pt", conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)

    def detect(self, frame):
        #The method will return a list of detections: [(label, confidence, (x1, y1, x2, y2))]
        detections = []

        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                label = self.model.names[int(box.cls[0])]
                
                # Extract and cast coordinates to integers
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append((label, conf, (x1, y1, x2, y2)))

        return detections