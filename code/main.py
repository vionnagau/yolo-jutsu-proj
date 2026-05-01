import cv2
import tkinter as tk
from PIL import Image, ImageTk, ImageFilter, ImageEnhance

import os
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from detector import JutsuDetector
from game_state import JutsuGame

from vfx_processor import clear_all_vfx, overlay_effect

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 600
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
SIGN_IMG_SIZE = 64
ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')

class NarutoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Naruto YOLOv8 Jutsu App")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg="#1e1e1e")

        # 1. LOGIC INITIALIZATION
        self.detector = JutsuDetector(model_path="../model/best.pt")
        self.game = JutsuGame()
        self.current_win_w = 900
        self.current_win_h = 600
        self.running = True
        self.mp_timestamp = 0

        # 2. UI SETUP (create self.video_canvas)
        self.setup_ui()

        # 3. CAMERA SETUP
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

        # STATE 1: Just the video feed and AI detection
        #self.running = True
        # (MOVE FROM HERE TO BELOW)
        #self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # LOOP 1
        # remove first loop to avoid running two loops at the same time, we will call the update_loop after setting up mediapipe in STATE 2
        # also to avoid running the heavy AI detection before we set up the hand tracking, which will help with better jutsu recognition and reduce false positives
        # prevent crashes and performance issues by only running one loop at a time
        #self.update_loop()
        
        # STATE 2: Add Mediapipe hand tracking for better jutsu recognition
        # (MOVE FROM HERE TO ABOVE)
        #self.running = True

        # 4. MEDIAPIPE SETUP
        hand_model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'hand_landmarker.task')
        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = HandLandmarker.create_from_options(hand_options)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # (MOVE FROM HERE TO ABOVE)
        #self.mp_timestamp = 0

        # LOOP 2
        #self.update_loop()
        
        # 5. START LOOP (ONLY after everything above is ready)
        # use after(100) to give Tkinter a bit moment to render the canvas
        self.root.after(100, self.update_loop)
        
    def setup_ui(self):
        # 1. DASHBOARD (Bottom Bar)
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
            text="Next: Hand Sign",
            font=("Verdana", 12, "bold"),
            fg="white", bg="#1a1a1a"
        )
        self.next_label.pack(pady=(0, 3))
        
        self.strip_frame = tk.Frame(self.dashboard, bg="#1a1a1a")
        self.strip_frame.pack(pady=2, expand=True)
        self.sign_widgets = []
        self.sign_images  = {}

        # 2. CAMERA CANVAS (MUST BE HERE, NOT IN REFRESH_STRIP)
        self.video_canvas = tk.Canvas(
            self.root,
            width=WINDOW_WIDTH, height=WINDOW_HEIGHT,
            bg="black", highlightthickness=0
        )
        self.video_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.video_canvas.bind("<Configure>", self.on_resize)

        # 3. SETTINGS BUTTON
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

        # 4. INITIALIZE THE FIRST SET OF ICONS IN THE STRIP
        self.refresh_strip()
     
    # start of the function that updates the hand sign strip at the bottom of the screen   
    def refresh_strip(self):
        # 1. Clear existing icons from the UI frame and reset state containers
        for widget in self.strip_frame.winfo_children():
            widget.destroy()

        self.sign_widgets = []
        self.sign_images.clear()

        status = self.game.get_status()
        sequence = status.get("sequence", [])
        current_idx = status.get("current_index", 0)

        for i, sign_name in enumerate(sequence):
            main_frame = tk.Frame(self.strip_frame, bg="#1a1a1a", padx=5)
            main_frame.pack(side=tk.LEFT)

            border_frame = tk.Frame(main_frame, bg="#1a1a1a", highlightthickness=2, highlightbackground="#1a1a1a")
            border_frame.pack()

            img_frame = tk.Frame(border_frame, bg="#1a1a1a")
            img_frame.pack(padx=2, pady=2)

            img_filename = f"{sign_name.lower()}.png"
            img_path = os.path.join(ASSETS_DIR, img_filename)

            try:
                pil_img = Image.open(img_path).convert("RGBA").resize((SIGN_IMG_SIZE, SIGN_IMG_SIZE), Image.LANCZOS)
            except Exception as e:
                print(f"UI Error: Could not load asset {img_path}: {e}")
                pil_img = Image.new("RGBA", (SIGN_IMG_SIZE, SIGN_IMG_SIZE), (40, 40, 40, 255))

            tk_img_normal = ImageTk.PhotoImage(pil_img)
            tk_img_dimmed = ImageTk.PhotoImage(ImageEnhance.Brightness(pil_img).enhance(0.45))

            normal_key = f"{sign_name}_{i}_normal"
            dimmed_key = f"{sign_name}_{i}_dimmed"
            self.sign_images[normal_key] = tk_img_normal
            self.sign_images[dimmed_key] = tk_img_dimmed

            lbl_img = tk.Label(img_frame, image=tk_img_normal, bg="#1a1a1a")
            lbl_img.image = tk_img_normal
            lbl_img.pack()

            lbl_text = tk.Label(img_frame, text=sign_name, fg="white", bg="#1a1a1a", font=("Arial", 8))
            lbl_text.pack()

            if i == current_idx:
                border_frame.config(highlightbackground="yellow", highlightthickness=2)
            else:
                border_frame.config(highlightthickness=2)

            self.sign_widgets.append({
                "lbl_img": lbl_img,
                "lbl_text": lbl_text,
                "border_frame": border_frame,
                "name": sign_name,
                "index": i
            })
        
    def on_resize(self, event):
        # This updates whenever the window is resized
        # Use event.width and event.height to get new dimensions
        self.current_win_w = event.width
        self.current_win_h = event.height
        
    def update_loop(self):
        # 1. Capture and Pre-process
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update_loop)
            return

        frame = cv2.flip(frame, 1) # Mirroring for user experience
        
        # ERROR FIX: Pull the dictionary from your game_state.py
        status = self.game.get_status() 
        detections = []
        target_x, target_y = None, None

        # --- LOGIC PHASE ---
        if not status["is_complete"]:
            # Active Detection: Looking for hand signs
            detections = self.detector.detect(frame)
            
            # ERROR FIX: Normalize labels to lowercase to match assets (e.g., bird.png)
            detected_labels = [d[0].lower().strip() for d in detections]
            
            # ERROR FIX: Use 'current_index' from your specific get_status() keys
            old_index = status.get("current_index", 0)
            
            # Update game logic
            self.game.update(detected_labels)

            # ERROR FIX: Check if progress was made, then refresh UI icons
            new_status = self.game.get_status()
            if new_status.get("current_index", 0) > old_index:
                self.refresh_strip()

            # Draw Bounding Boxes
            for label, conf, (x1, y1, x2, y2) in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # VFX Mode: Track hand for effect placement
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            self.mp_timestamp += 33 
            results = self.hand_landmarker.detect_for_video(mp_image, self.mp_timestamp)
            
            if results.hand_landmarks:
                lm = results.hand_landmarks[0][9] # Middle finger MCP
                target_x = int(lm.x * 1280) # Use your VIDEO_WIDTH
                target_y = int(lm.y * 720)  # Use your VIDEO_HEIGHT
            
            if target_x is not None:
                # ERROR FIX: Pass status["target"].lower() to find frame folders
                vfx_result = overlay_effect(frame, status["target"].lower(), target_x, target_y, 
                                            size=400, sw=1280, sh=720)
                frame = vfx_result['frame']
                
                if vfx_result['effect_finished']:
                    self.game.is_effect_complete = True
                    self.game.update([]) # Reset for new round
                    self.refresh_strip()

        # --- UI DRAWING ON FRAME ---
        cv2.putText(frame, f"TARGET: {status['target'].upper()}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # ERROR FIX: Use 'next_sign' from your get_status() to avoid AttributeError
        guidance_text = f"NEXT: {status['next_sign'].upper()}"
        text_color = (0, 255, 0) if status["is_complete"] else (255, 255, 255)
        
        cv2.putText(frame, guidance_text, (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        # --- RENDERING PHASE ---
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()
        if canvas_w < 2: canvas_w = 1280 
        if canvas_h < 2: canvas_h = 720  

        display_frame = cv2.resize(frame, (canvas_w, canvas_h))
        img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))

        if not hasattr(self, 'video_image_id'):
            self.video_image_id = self.video_canvas.create_image(0, 0, image=tk_img, anchor=tk.NW)
        else:
            self.video_canvas.itemconfig(self.video_image_id, image=tk_img)

        self.video_canvas.image = tk_img 
        self.update_dashboard()
        self.root.after(30, self.update_loop)

    # Clean up resources to prevent memory leaks or camera lock-up
    def on_closing(self):
        self.cap.release()
        self.root.destroy()
        
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

            for widget in self.sign_widgets:
                lbl_img = widget["lbl_img"]
                lbl_text = widget["lbl_text"]
                border_frame = widget["border_frame"]
                name = widget["name"]
                idx = widget["index"]

                normal_key = f"{name}_{idx}_normal"
                dimmed_key = f"{name}_{idx}_dimmed"
                normal_img = self.sign_images.get(normal_key)
                dimmed_img = self.sign_images.get(dimmed_key, normal_img)

                if idx < current_idx:
                    lbl_img.config(image=dimmed_img)
                    lbl_img.image = dimmed_img
                    border_frame.config(bg="#00ff00")
                    lbl_text.config(fg="#888")
                elif idx == current_idx and not is_complete:
                    lbl_img.config(image=normal_img)
                    lbl_img.image = normal_img
                    border_frame.config(bg="#ffaa00")
                    lbl_text.config(fg="#ffaa00")
                else:
                    lbl_img.config(image=normal_img)
                    lbl_img.image = normal_img
                    border_frame.config(bg="#1a1a1a")
                    lbl_text.config(fg="white")

        except Exception as e:
            print(f"UI Error: {e}")
            
    def cycle_jutsu(self):
        # clear any fire/lightning effects currently on screen
        clear_all_vfx()
        
        # tell the game logic to move to the next Jutsu in the list
        self.game.next_jutsu()
        
        # re-draw the hand sign icons at the bottom for the new jutsu
        self.refresh_strip()
        
        # update the text labels (ex: CHIDORI, etc) immediately
        self.update_dashboard()

if __name__ == "__main__":
    root = tk.Tk()
    app = NarutoApp(root)
    root.mainloop()
