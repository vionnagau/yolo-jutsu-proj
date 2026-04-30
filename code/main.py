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
        # This creates the dark gray bar at the bottom of your window
        self.dashboard = tk.Frame(self.root, bg="#1a1a1a")
        self.dashboard.pack(side=tk.BOTTOM, fill=tk.X)
        
        # show the Jutsu Name like CHIDORI
        self.target_label = tk.Label(
            self.dashboard,
            text="CHIDORI",
            font=("Verdana", 8, "bold"),
            fg="#888", bg="#1a1a1a"
        )
        self.target_label.pack(pady=(2, 0))
        
        # tell the user which hand sign to do next
        self.next_label = tk.Label(
            self.dashboard,
            text="Next: Hand Sign",
            font=("Verdana", 12, "bold"),
            fg="white", bg="#1a1a1a"
        )
        self.next_label.pack(pady=(0, 3))
        
        # this is where the small hand-sign images will sit
        self.strip_frame = tk.Frame(self.dashboard, bg="#1a1a1a")
        self.strip_frame.pack(pady=2, expand=True)
        self.sign_widgets = []
        self.sign_images  = {}
        
        # WARNING
        #self.refresh_strip()
     
    # start of the function that updates the hand sign strip at the bottom of the screen   
    def refresh_strip(self):
        # clear out any existing hand sign pictures first
        for w in self.sign_widgets:
            w["main_frame"].destroy() 
        self.sign_widgets = []
        
        # get the new jutsu sequence (ex:'Dog', 'Bird', 'Monkey')
        status = self.game.get_status()
        sequence = status["sequence"]
        SIGN_IMG_SIZE = 55
        
        for i, sign_name in enumerate(sequence):
            # create a small container for each hand sign
            main_frame = tk.Frame(self.strip_frame, bg="#1a1a1a", padx=10)
            main_frame.pack(side=tk.LEFT)
            
            border_frame = tk.Frame(main_frame, bg="#1a1a1a", padx=3, pady=3)
            border_frame.pack()
            
            # find the image in assets folder
            img_path = os.path.join(os.path.dirname(__file__), '..', 'assets', f'{sign_name.lower()}.png')
            pil_img = None
            
            if os.path.exists(img_path):
                try:
                    pil_img = Image.open(img_path).convert("RGBA").resize((SIGN_IMG_SIZE, SIGN_IMG_SIZE))
                except Exception as e:
                    print(f"Failed to load image: {e}")
            
            # if the image is missing, create a gray square so the app doesn't crash
            if pil_img is None:
                pil_img = Image.new('RGBA', (SIGN_IMG_SIZE, SIGN_IMG_SIZE), color=(50, 50, 50, 255))
            
            tk_img = ImageTk.PhotoImage(pil_img)
            
            # create the "Dimmed" version (blurred and dark) for signs not yet completed
            try:
                blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=8))
                enhancer = ImageEnhance.Brightness(blurred)
                darkened = enhancer.enhance(0.5)
                tk_img_dimmed = ImageTk.PhotoImage(darkened)
            except Exception as e:
                print(f"Effect Error: {e}")
                tk_img_dimmed = tk_img 

            # save the images in a dictionary so they don't get deleted by Python's memory manager
            self.sign_images[f"{sign_name}_{i}_normal"] = tk_img 
            self.sign_images[f"{sign_name}_{i}_dimmed"] = tk_img_dimmed
            
            # put the image on the screen
            lbl_img = tk.Label(border_frame, image=tk_img, bg="#1a1a1a", bd=0)
            lbl_img.pack()
            
            # put the text (name of the sign) under the image
            lbl_text = tk.Label(main_frame, text=sign_name, font=("Arial", 9, "bold"), fg="white", bg="#1a1a1a")
            lbl_text.pack(pady=2)
            
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
        
        # camera feed canvas
        self.video_canvas = tk.Canvas(
            self.root,
            width=VIDEO_WIDTH, height=VIDEO_HEIGHT,
            bg="black", highlightthickness=0
        )
        self.video_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # floating button to change jutsus
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
            
            
        # START - ADD: VFX LOGIC
        
        # draw VFX if the jutsu sequence is finished
        if status["is_complete"]:
            # Check if the AI can actually see your hand (target_x and target_y)
            hand_visible = target_x is not None and target_y is not None

            if hand_visible:
                # draws the animation (fireball/lightning) on the hand
                vfx_result = overlay_effect(
                    frame, status["target"], target_x, target_y,
                    size=350, sw=VIDEO_WIDTH, sh=VIDEO_HEIGHT
                )
                
                # tell the game if the animation has finished playing
                self.game.is_effect_complete = vfx_result['effect_finished']
                frame = vfx_result['frame']

                # if the fireball has disappeared, reset the game state
                if vfx_result['effect_finished']:
                    self.game.update([])

            # show a big red "ACTIVATED!" message at the bottom of the video
            msg = f"{status['target']}: ACTIVATED!"
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            tx = (VIDEO_WIDTH - tw) // 2
            cv2.putText(frame, msg, (tx, VIDEO_HEIGHT - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # END - ADD: VFX LOGIC


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

                # REPLACEMENT
                # # 4. DISPLAY
                # # Convert from Tkinter
                # img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # pil_img = Image.fromarray(img_rgb)
                # tk_img = ImageTk.PhotoImage(image=pil_img)

                # # 5. RENDER TO CANVAS
                # if not hasattr(self, 'video_image_id'):
                #     self.video_image_id = self.video_canvas.create_image(
                #         self.current_win_w // 2, self.current_win_h // 2, 
                #         image=tk_img, anchor=tk.CENTER
                #     )
                # else:
                #     self.video_canvas.itemconfig(self.video_image_id, image=tk_img)
                #     self.video_canvas.coords(self.video_image_id, 
                #                            self.current_win_w // 2, 
                #                            self.current_win_h // 2)

                # self.video_canvas.image = tk_img
                
                # REPLACE (START)
                # Convert to Tkinter
                canvas_w = self.video_canvas.winfo_width()
                canvas_h = self.video_canvas.winfo_height()

                # Fast OpenCV resize instead of slow PIL resize
                # This checks if the window size has changed from the original VIDEO_WIDTH/HEIGHT
                if canvas_w > 1 and canvas_h > 1 and (canvas_w != VIDEO_WIDTH or canvas_h != VIDEO_HEIGHT):
                    display_frame = cv2.resize(frame, (canvas_w, canvas_h))
                else:
                    display_frame = frame
                    
                img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                tk_img = ImageTk.PhotoImage(image=pil_img)

                # Update Canvas Image
                if not hasattr(self, 'video_image_id'):
                    # Note: anchor=tk.NW puts the image at the top-left corner
                    self.video_image_id = self.video_canvas.create_image(0, 0, image=tk_img, anchor=tk.NW)
                else:
                    self.video_canvas.itemconfig(self.video_image_id, image=tk_img)

                self.video_canvas.image = tk_img 
                # REPLACE (END)
                
                # ADD: CALL THE FUNCTION TO UPDATE THE HAND SIGN STRIP
                # Update the dashboard with the current game status
                self.update_dashboard()

        # 6. REPEAT EVERY 30ms
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
