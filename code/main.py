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
        
        #FIX: remove canvas and button initialization from this function
        #this function now ONLY handle the refreshing of hand-sign icons
        
        # clear out any existing hand sign pictures first
        for w in self.sign_widgets:
            w["main_frame"].destroy() 
        self.sign_widgets = []
        
        # get the new jutsu sequence (ex:'Dog', 'Bird', 'Monkey')
        status = self.game.get_status()
        sequence = status["sequence"]
        SIGN_IMG_SIZE = 55
        
        for i, sign_name in enumerate(sequence):
            
            # ERROR FIX: Ensure frames are child of self.strip_frame
            # create a small container for each hand sign
            main_frame = tk.Frame(self.strip_frame, bg="#1a1a1a", padx=5)
            main_frame.pack(side=tk.LEFT)

            border_frame = tk.Frame(main_frame, bg="#333", padx=2, pady=2)
            border_frame.pack()
            
            # Load and cache image
            img_path = f"assets/signs/{sign_name}.png"
            try:
                pil_img = Image.open(img_path).resize((SIGN_IMG_SIZE, SIGN_IMG_SIZE))
                tk_img = ImageTk.PhotoImage(pil_img)
                self.sign_images[f"{sign_name}_{i}"] = tk_img 
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                tk_img = None

            lbl_img = tk.Label(border_frame, image=tk_img, bg="black")
            lbl_img.pack()

            lbl_text = tk.Label(main_frame, text=sign_name.upper(), 
                               fg="#888", bg="#1a1a1a", font=("Arial", 7, "bold"))
            lbl_text.pack()
            
            # keep track of everything so can change the colors later
            self.sign_widgets.append({
                "main_frame": main_frame,
                "border_frame": border_frame,
                "lbl_img": lbl_img, 
                "lbl_text": lbl_text,
                "name": sign_name, 
                "id": i
            })
            
        # this will update the text labels (ex: CHIDORI, etc)
        self.update_dashboard()
        
        # end of the function that updates the hand sign strip at the bottom of the screen
        
        # WARNING
        # video_canvas and settings_btn MUST be moved to setup_ui() 
        # so they are not destroyed/recreated every time the jutsu changes
        
        # # camera feed canvas
        # self.video_canvas = tk.Canvas(
        #     self.root,
        #     width=VIDEO_WIDTH, height=VIDEO_HEIGHT,
        #     bg="black", highlightthickness=0
        # )
        # self.video_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # # floating button to change jutsus
        # self.settings_btn = tk.Button(
        #     self.root,
        #     text="NEXT JUTSU >>",
        #     font=("Arial", 12, "bold"),
        #     bg="#444", fg="white",
        #     activebackground="#666", activeforeground="white",
        #     bd=0, padx=15, pady=8,
        #     command=self.cycle_jutsu
        # )
        # self.settings_btn.place(x=WINDOW_WIDTH - 200, y=20)
        
    def on_resize(self, event):
        # This updates whenever the window is resized
        # Use event.width and event.height to get new dimensions
        self.current_win_w = event.width
        self.current_win_h = event.height
        
    def update_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update_loop)
            return

        frame = cv2.flip(frame, 1) # Mirroring
        status = self.game.get_status()
        detections = []
        target_x, target_y = None, None

        # --- LOGIC PHASE ---
        if not status["is_complete"]:
            # Active Detection Mode
            detections = self.detector.detect(frame)
            detected_labels = [d[0].replace("-", "").strip() for d in detections]
            self.game.update(detected_labels)

            for label, conf, (x1, y1, x2, y2) in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # VFX/Hand Tracking Mode
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            self.mp_timestamp += 33 
            results = self.hand_landmarker.detect_for_video(mp_image, self.mp_timestamp)
            
            if results.hand_landmarks:
                lm = results.hand_landmarks[0][9] # Middle finger MCP
                target_x = int(lm.x * VIDEO_WIDTH)
                target_y = int(lm.y * VIDEO_HEIGHT)
            
            if target_x is not None:
                vfx_result = overlay_effect(frame, status["target"], target_x, target_y, 
                                          size=400, sw=VIDEO_WIDTH, sh=VIDEO_HEIGHT)
                frame = vfx_result['frame']
                if vfx_result['effect_finished']:
                    self.game.is_effect_complete = True
                    self.game.update([]) # Resets the game

        # --- UI DRAWING ON FRAME ---
        cv2.putText(frame, f"TARGET: {status['target'].upper()}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # --- RENDERING PHASE (Must be outside the IF blocks) ---
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()

        # Fallback for initial startup
        if canvas_w < 2: canvas_w = WINDOW_WIDTH
        if canvas_h < 2: canvas_h = WINDOW_HEIGHT

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
            
            # update the main text labels at the bottom
            self.target_label.config(text=f"TARGET: {status['target'].upper()}")
            
            if is_complete:
                self.next_label.config(text="JUTSU ACTIVATED!", fg="#00ffcc")
                self.target_label.config(fg="#00ffcc")
            else:
                self.next_label.config(text=f"NEXT: {status['next_sign']}", fg="white")
                self.target_label.config(fg="#888")

            # loop through each hand sign icon in the bottom bar
            for i, widget in enumerate(self.sign_widgets):
                lbl_img = widget["lbl_img"]
                lbl_text = widget["lbl_text"]
                border_frame = widget["border_frame"]
                name = widget["name"]
                
                if i < current_idx:
                    # DONE: If finished this sign, give a Green Border
                    lbl_img.config(image=self.sign_images[f"{name}_{i}_dimmed"])
                    border_frame.config(bg="#00ff00") 
                    lbl_text.config(fg="#888") 
                    
                elif i == current_idx and not is_complete:
                    # ACTIVE: The sign that is needed to do right now gets an Orange Border
                    lbl_img.config(image=self.sign_images[f"{name}_{i}_normal"])
                    border_frame.config(bg="#ffaa00") 
                    lbl_text.config(fg="#ffaa00")
                    
                else:
                    # PENDING: Signs that haven't reached yet stay dark
                    lbl_img.config(image=self.sign_images[f"{name}_{i}_normal"])
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
