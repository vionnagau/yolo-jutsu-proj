import cv2
import tkinter as tk
import os
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from detector import JutsuDetector
from game_state import JutsuGame
from vfx_processor import clear_all_vfx, overlay_effect

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 600 
VIDEO_WIDTH = 900
VIDEO_HEIGHT = 450

class NarutoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Naruto YOLOv8 Jutsu App")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg="#1e1e1e")

        # LOGIC INITIALIZATION
        self.detector = JutsuDetector(model_path="../model/best.pt")
        self.game = JutsuGame()
        
        self.setup_ui()

        # CAMERA SETUP
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

        # STATE
        self.running = True

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

        #LOOP
        self.update_loop()

    def setup_ui(self):
        self.dashboard = tk.Frame(self.root, bg="#1a1a1a")
        self.dashboard.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.target_label = tk.Label(
            self.dashboard,
            text="CHIDORI",
            font=("Verdana", 8, "bold"),
            fg="#888", bg="#1a1a1a"
        )
        self.target_label.pack(pady=(2, 0))
        
        self.next_label = tk.Label(
            self.dashboard,
            text="Next: OX",
            font=("Verdana", 12, "bold"),
            fg="white", bg="#1a1a1a"
        )
        self.next_label.pack(pady=(0, 3))
        
        self.strip_frame = tk.Frame(self.dashboard, bg="#1a1a1a")
        self.strip_frame.pack(pady=2, expand=True)
        self.sign_widgets = []
        self.sign_images  = {}
        self.refresh_strip()
        
        self.video_canvas = tk.Canvas(
            self.root,
            width=VIDEO_WIDTH, height=VIDEO_HEIGHT,
            bg="black", highlightthickness=0
        )
        self.video_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.settings_btn = tk.Button(
            self.root,
            text="NEXT JUTSU >>",
            font=("Arial", 12, "bold"),
            bg="#444", fg="white",
            activebackground="#666", activeforeground="white",
            bd=0, padx=15, pady=8,
            command=self.cycle_jutsu
        )
        self.settings_btn.place(x=WINDOW_WIDTH - 200, y=20)

    def refresh_strip(self):
        for w in self.sign_widgets:
            w["main_frame"].destroy() 
        self.sign_widgets = []
        
        status = self.game.get_status()
        sequence = status["sequence"]
        SIGN_IMG_SIZE = 55
        
        for i, sign_name in enumerate(sequence):
            main_frame = tk.Frame(self.strip_frame, bg="#1a1a1a", padx=10)
            main_frame.pack(side=tk.LEFT)
            
            border_frame = tk.Frame(main_frame, bg="#1a1a1a", padx=3, pady=3)
            border_frame.pack()
            
            img_path = os.path.join(os.path.dirname(__file__), '..', 'assets', f'{sign_name.lower()}.png')
            pil_img = None
            
            if os.path.exists(img_path):
                try:
                    pil_img = Image.open(img_path).convert("RGBA").resize((SIGN_IMG_SIZE, SIGN_IMG_SIZE))
                except Exception as e:
                    print(f"Failed to load image: {e}")
            
            if pil_img is None:
                pil_img = Image.new('RGBA', (SIGN_IMG_SIZE, SIGN_IMG_SIZE), color=(50, 50, 50, 255))
            
            tk_img = ImageTk.PhotoImage(pil_img)
            
            try:
                blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=8))
                enhancer = ImageEnhance.Brightness(blurred)
                darkened = enhancer.enhance(0.5)
                tk_img_dimmed = ImageTk.PhotoImage(darkened)
            except Exception as e:
                print(f"Effect Error: {e}")
                tk_img_dimmed = tk_img 

            self.sign_images[f"{sign_name}_{i}_normal"] = tk_img 
            self.sign_images[f"{sign_name}_{i}_dimmed"] = tk_img_dimmed
            lbl_img = tk.Label(border_frame, image=tk_img, bg="#1a1a1a", bd=0)
            lbl_img.pack()
            
            lbl_text = tk.Label(main_frame, text=sign_name, font=("Arial", 9, "bold"), fg="white", bg="#1a1a1a")
            lbl_text.pack(pady=2)
            self.sign_widgets.append({
                "main_frame": main_frame,
                "border_frame": border_frame,
                "lbl_img": lbl_img, 
                "lbl_text": lbl_text,
                "name": sign_name, 
                "id": i
            })
            
        self.update_dashboard()

    def update_dashboard(self):
        try:
            status = self.game.get_status()
            current_idx = status["current_index"]
            is_complete = status["is_complete"]
            
            self.target_label.config(text=f"TARGET: {status['target'].upper()}")
            
            if is_complete:
                self.next_label.config(text="JUTSU ACTIVATED!", fg="#00ffcc")
                self.target_label.config(fg="#00ffcc")
            else:
                self.next_label.config(text=f"NEXT: {status['next_sign']}", fg="white")
                self.target_label.config(fg="#888")

            for i, widget in enumerate(self.sign_widgets):
                lbl_img = widget["lbl_img"]
                lbl_text = widget["lbl_text"]
                border_frame = widget["border_frame"]
                name = widget["name"]
                
                if i < current_idx:
                    # DONE: Green Border + Dimmed
                    lbl_img.config(image=self.sign_images[f"{name}_{i}_dimmed"])
                    border_frame.config(bg="#00ff00") 
                    lbl_text.config(fg="#888") 
                    
                elif i == current_idx and not is_complete:
                    # ACTIVE: ORANGE Border + Normal
                    lbl_img.config(image=self.sign_images[f"{name}_{i}_normal"])
                    border_frame.config(bg="#ffaa00") 
                    lbl_text.config(fg="#ffaa00")
                    
                else:
                    # PENDING: Dark Border + Normal
                    lbl_img.config(image=self.sign_images[f"{name}_{i}_normal"])
                    border_frame.config(bg="#1a1a1a") # Matches background
                    lbl_text.config(fg="white")
        except Exception as e:
            print(f"UI Error: {e}")

    def update_loop(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1) # mirroring our camera
            raw_h, raw_w = frame.shape[:2]

            # We're going to do some basic math to match our WINDOW_WIDTH
            # And then we'll match the height based on the aspect ratio (scale_factor)
            scale_factor = VIDEO_WIDTH / raw_w
            new_h = int(raw_h * scale_factor)
            frame = cv2.resize(frame, (VIDEO_WIDTH, new_h))
            frame = frame[0:VIDEO_HEIGHT, 0:VIDEO_WIDTH]

            status = self.game.get_status()
            target_x, target_y = None, None # We're initializing these coordinates for later
            detections = []

            # We're checking "Are we still playing, or are we done?"
            if not status["is_complete"]:
                detections = self.detector.detect(frame)
                detected_labels = [d[0].replace("-", "").strip() for d in detections]
                self.game.update(detected_labels)
                
                # Drawing bounding boxes
                for label, conf, (x1, y1, x2, y2) in detections:
                    color = (0, 255, 0) # Green in BGR
                    cv2. rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    text = f"{label}: {conf:.2f}"
                    (w, h), _  = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                    cv2.putText(
                        frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
            # After Jutsu has been detected, we switch to VFX Mode!
            else:
                # Run MediaPipe Hands
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                self.mp_timestamp += 33 
                results = self.hand_landmarker.detect_for_video(mp_image, self.mp_timestamp)
                
                if results.hand_landmarks:
                    hand_landmarks = results.hand_landmarks[0]
                    lm = hand_landmarks[9]
                    target_x = int(lm.x * VIDEO_WIDTH)
                    target_y = int(lm.y * VIDEO_HEIGHT)

            cv2.putText(frame, "CURRENT TARGET:", (VIDEO_WIDTH - 500, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(frame, status['target'].upper(), (VIDEO_WIDTH - 500, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
            
            # Draw VFX if complete
            if status["is_complete"]:
                hand_visible = target_x is not None and target_y is not None

                if hand_visible:
                    vfx_result = overlay_effect(
                        frame, status["target"], target_x, target_y,
                        size=350, sw=VIDEO_WIDTH, sh=VIDEO_HEIGHT
                    )
                    self.game.is_effect_complete = vfx_result['effect_finished']
                    frame = vfx_result['frame']

                    if vfx_result['effect_finished']:
                        self.game.update([])

                msg = f"{status['target']}: ACTIVATED!"
                (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                tx = (VIDEO_WIDTH - tw) // 2
                cv2.putText(frame, msg, (tx, VIDEO_HEIGHT - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        
            # Convert to Tkinter
            canvas_w = self.video_canvas.winfo_width()
            canvas_h = self.video_canvas.winfo_height()
            
            # Fast OpenCV resize instead of slow PIL Image.LANCZOS
            if canvas_w > 1 and canvas_h > 1 and (canvas_w != VIDEO_WIDTH or canvas_h != VIDEO_HEIGHT):
                display_frame = cv2.resize(frame, (canvas_w, canvas_h))
            else:
                display_frame = frame
                
            img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tk_img = ImageTk.PhotoImage(image=pil_img)
            
            # Update Canvas Image
            if not hasattr(self, 'video_image_id'):
                 self.video_image_id = self.video_canvas.create_image(0, 0, image=tk_img, anchor=tk.NW)
            else:
                 self.video_canvas.itemconfig(self.video_image_id, image=tk_img)
            
            self.video_canvas.image = tk_img 
            
            # 8. UI Updates
            self.update_dashboard()

        # We're calling this function to update the frame every 30ms
        self.root.after(30, self.update_loop)
    
    def cycle_jutsu(self):
        clear_all_vfx()
        self.game.next_jutsu()
        self.refresh_strip()
        self.update_dashboard()

if __name__ == "__main__":
    root = tk.Tk()
    app = NarutoApp(root)
    root.mainloop()