import cv2
import tkinter as tk
from PIL import Image, ImageTk, ImageFilter, ImageEnhance

import os
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from detector import JutsuDetector
from game_state import JutsuGame

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 600
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
class NarutoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Naruto YOLOv8 Jutsu App")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg="#1e1e1e")

        # LOGIC INITIALIZATION
        self.detector = JutsuDetector(model_path="../model/best.pt")
        self.game = JutsuGame()
        
        self.current_win_w = 900
        self.current_win_h = 600

        self.setup_ui()

        # CAMERA SETUP
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

        # STATE 1: Just the video feed and AI detection
        #self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # LOOP 1
        # remove first loop to avoid running two loops at the same time, we will call the update_loop after setting up mediapipe in STATE 2
        # also to avoid running the heavy AI detection before we set up the hand tracking, which will help with better jutsu recognition and reduce false positives
        # prevent crashes and performance issues by only running one loop at a time
        #self.update_loop()
        
        # STATE 2: Add Mediapipe hand tracking for better jutsu recognition
        self.running = True

        # MEDIAPIPE SETUP
        hand_model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'hand_landmarker.task')
        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = HandLandmarker.create_from_options(hand_options)
        self.mp_timestamp = 0

        # LOOP 2
        self.update_loop()
        
    def setup_ui(self):
        self.video_canvas = tk.Canvas(
            self.root,
            #width=VIDEO_WIDTH,
            #height=VIDEO_HEIGHT,
            bg="black",
            highlightthickness=0
        )
        self.video_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # IMPORTANT: This line connects the window resizing to the app
        self.root.bind("<Configure>", self.on_resize)
        
    def on_resize(self, event):
        # This updates whenever the window is resized
        # Use event.width and event.height to get new dimensions
        self.current_win_w = event.width
        self.current_win_h = event.height
        
    def update_loop(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1) # mirroring our camera
            #raw_h, raw_w = frame.shape[:2]
            
            # STATE-AWARE LOGIC (START)
            status = self.game.get_status()
            target_x, target_y = None, None 
            detections = []

            # Only run the heavy AI detector if the jutsu is NOT complete
            if not status["is_complete"]:
                detections = self.detector.detect(frame)
                detected_labels = [d[0].replace("-", "").strip() for d in detections]
                self.game.update(detected_labels)
                
                # Drawing bounding boxes (Moved inside this 'if' block)
                for label, conf, (x1, y1, x2, y2) in detections:
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    text = f"{label}: {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    # This adds a nice solid background to the text label
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                    cv2.putText(
                        frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
            # STATE-AWARE LOGIC (END)
            
            # VFX Path
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                self.mp_timestamp += 33 
                results = self.hand_landmarker.detect_for_video(mp_image, self.mp_timestamp)
                
                if results.hand_landmarks:
                    hand_landmarks = results.hand_landmarks[0]
                    lm = hand_landmarks[9]
                    target_x = int(lm.x * VIDEO_WIDTH)
                    target_y = int(lm.y * VIDEO_HEIGHT)

            # UI & DISPLAY (START)
            # Drawing the target jutsu name on the video frame
            # (outside if block so it always shows UI)
            status = self.game.get_status()
            cv2.putText(frame, "CURRENT TARGET:", (VIDEO_WIDTH - 500, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(frame, status['target'].upper(), (VIDEO_WIDTH - 500, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)


            # 2. Drawing the bounding boxes
            for label, conf, (x1, y1, x2, y2) in detections:
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 3. SCALING THE FRAME TO FIT (MIN) / FILL (MAX)
            raw_h, raw_w = frame.shape[:2]
            # Calculate scale based on the variables updated by on_resize
            scale = max(self.current_win_w / raw_w, self.current_win_h / raw_h)
            
            if scale > 0:
                new_w, new_h = int(raw_w * scale), int(raw_h * scale)
                # cv2.INTER_AREA >> best for shrinking
                # cv2.INTER_CUBIC >> best for enlarging
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

                # 4. DISPLAY
                # Convert from Tkinter
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                tk_img = ImageTk.PhotoImage(image=pil_img)

                # 5. RENDER TO CANVAS
                if not hasattr(self, 'video_image_id'):
                    self.video_image_id = self.video_canvas.create_image(
                        self.current_win_w // 2, self.current_win_h // 2, 
                        image=tk_img, anchor=tk.CENTER
                    )
                else:
                    self.video_canvas.itemconfig(self.video_image_id, image=tk_img)
                    self.video_canvas.coords(self.video_image_id, 
                                           self.current_win_w // 2, 
                                           self.current_win_h // 2)

                self.video_canvas.image = tk_img

        # 6. REPEAT EVERY 30ms
        self.root.after(30, self.update_loop)

    # Clean up resources to prevent memory leaks or camera lock-up
    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = NarutoApp(root)
    root.mainloop()
