import cv2
import tkinter as tk
from PIL import Image, ImageTk, ImageEnhance

import os
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from detector import JutsuDetector
from game_state import JutsuGame
from vfx_processor import clear_all_vfx, overlay_effect

WINDOW_WIDTH  = 900
WINDOW_HEIGHT = 600
VIDEO_WIDTH   = 1280
VIDEO_HEIGHT  = 720
SIGN_IMG_SIZE = 64
ASSETS_DIR    = os.path.join(os.path.dirname(__file__), '..', 'assets')


class NarutoApp:
    """
    main application class.
    wires together: tkinter ui, opencv camera, yolov8 detector,
    mediapipe hand tracker, game logic, and vfx renderer.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Naruto Jutsu — Real-Time Hand Sign Recognition")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg="#1e1e1e")

        # core components
        self.detector = JutsuDetector()  # model_path is auto-resolved inside detector.py
        self.game = JutsuGame()

        # canvas size (updates on window resize)
        self.current_win_w = WINDOW_WIDTH
        self.current_win_h = WINDOW_HEIGHT

        # mediapipe needs a monotonically increasing timestamp in milliseconds
        self.mp_timestamp = 0

        # build all tkinter widgets
        self.setup_ui()

        # open webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

        # load the mediapipe hand landmark model
        hand_model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'hand_landmarker.task')
        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_landmarker = HandLandmarker.create_from_options(hand_options)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # give tkinter 100ms to render the canvas before starting the loop
        self.root.after(100, self.update_loop)

    # -------------------------------------------------------------------------
    # ui setup
    # -------------------------------------------------------------------------

    def setup_ui(self):
        # bottom dashboard bar (target label, next sign label, icon strip)
        self.dashboard = tk.Frame(self.root, bg="#1a1a1a")
        self.dashboard.pack(side=tk.BOTTOM, fill=tk.X)

        self.target_label = tk.Label(self.dashboard, text="", font=("Verdana", 8, "bold"), fg="#888", bg="#1a1a1a")
        self.target_label.pack(pady=(2, 0))

        self.next_label = tk.Label(self.dashboard, text="NEXT: Hand Sign", font=("Verdana", 12, "bold"), fg="white", bg="#1a1a1a")
        self.next_label.pack(pady=(0, 3))

        self.strip_frame = tk.Frame(self.dashboard, bg="#1a1a1a")
        self.strip_frame.pack(pady=2, expand=True)

        self.sign_widgets = []
        self.sign_images = {}

        # camera canvas — sits above the dashboard, fills remaining space
        self.video_canvas = tk.Canvas(self.root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg="black", highlightthickness=0)
        self.video_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.video_canvas.bind("<Configure>", self.on_resize)

        # button to skip to the next jutsu
        self.settings_btn = tk.Button(
            self.root, text="NEXT JUTSU >>",
            font=("Arial", 12, "bold"),
            bg="#444", fg="white",
            activebackground="#666", activeforeground="white",
            bd=0, padx=15, pady=8,
            command=self.cycle_jutsu,
        )
        self.settings_btn.place(x=WINDOW_WIDTH - 200, y=20)

        self.refresh_strip()

    def refresh_strip(self):
        """
        rebuild the icon row at the bottom of the screen.
        called once at startup and again whenever the active sign changes.
        """
        for widget in self.strip_frame.winfo_children():
            widget.destroy()

        self.sign_widgets = []
        self.sign_images.clear()

        status = self.game.get_status()
        sequence = status["sequence"]
        current_idx = status["current_index"]

        for i, sign_name in enumerate(sequence):
            # outer frame for spacing
            main_frame = tk.Frame(self.strip_frame, bg="#1a1a1a", padx=5)
            main_frame.pack(side=tk.LEFT)

            # coloured border highlights the active sign
            border_frame = tk.Frame(main_frame, bg="#1a1a1a", highlightthickness=2, highlightbackground="#1a1a1a")
            border_frame.pack()

            img_frame = tk.Frame(border_frame, bg="#1a1a1a")
            img_frame.pack(padx=2, pady=2)

            # load the hand sign image from assets/
            img_path = os.path.join(ASSETS_DIR, f"{sign_name.lower()}.png")
            try:
                pil_img = Image.open(img_path).convert("RGBA").resize((SIGN_IMG_SIZE, SIGN_IMG_SIZE), Image.LANCZOS)
            except Exception:
                pil_img = Image.new("RGBA", (SIGN_IMG_SIZE, SIGN_IMG_SIZE), (40, 40, 40, 255))

            # keep both a normal and dimmed version — dimmed = already completed
            tk_normal = ImageTk.PhotoImage(pil_img)
            tk_dimmed = ImageTk.PhotoImage(ImageEnhance.Brightness(pil_img).enhance(0.45))

            self.sign_images[f"{sign_name}_{i}_normal"] = tk_normal
            self.sign_images[f"{sign_name}_{i}_dimmed"] = tk_dimmed

            lbl_img = tk.Label(img_frame, image=tk_normal, bg="#1a1a1a")
            lbl_img.image = tk_normal
            lbl_img.pack()

            lbl_text = tk.Label(img_frame, text=sign_name, fg="white", bg="#1a1a1a", font=("Arial", 8))
            lbl_text.pack()

            if i == current_idx:
                border_frame.config(highlightbackground="yellow", highlightthickness=2)

            self.sign_widgets.append({
                "lbl_img": lbl_img, "lbl_text": lbl_text,
                "border_frame": border_frame, "name": sign_name, "index": i,
            })

    def on_resize(self, event):
        self.current_win_w = event.width
        self.current_win_h = event.height

    # -------------------------------------------------------------------------
    # main loop
    # -------------------------------------------------------------------------

    def update_loop(self):
        """
        runs every 30ms.
        phase 1 (jutsu not complete): run yolo on each frame to detect hand signs.
        phase 2 (jutsu complete):     run mediapipe to track hand position and play vfx.
        """
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update_loop)
            return

        # mirror the image so it feels like looking in a mirror
        frame = cv2.flip(frame, 1)

        status = self.game.get_status()

        # --- phase 1: sign detection ---
        if not status["is_complete"]:
            detections = self.detector.detect(frame)
            detected_labels = [d[0] for d in detections]  # labels already cleaned by detector
            print(f"\n[FRAME] Detected: {detected_labels}")

            old_index = status["current_index"]
            self.game.update(detected_labels)

            # refresh icons only when the step actually advanced
            new_status = self.game.get_status()
            if new_status["current_index"] > old_index:
                print(f"[FRAME] Sequence advanced! Old={old_index}, New={new_status['current_index']}")
                self.refresh_strip()

            # draw green bounding boxes + label for each detected sign
            for label, conf, (x1, y1, x2, y2) in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- phase 2: vfx playback ---
        else:
            # use mediapipe to find where the hand is so the effect can follow it
            hand_x, hand_y = None, None

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            self.mp_timestamp += 33
            results = self.hand_landmarker.detect_for_video(mp_image, self.mp_timestamp)

            if results.hand_landmarks:
                # landmark 9 = middle finger MCP — a stable central point on the hand
                lm = results.hand_landmarks[0][9]
                hand_x = int(lm.x * VIDEO_WIDTH)
                hand_y = int(lm.y * VIDEO_HEIGHT)

            # always call overlay_effect — if no hand detected it defaults to screen centre
            vfx_result = overlay_effect(
                frame, status["target"],
                hand_x, hand_y,
                size=400, sw=VIDEO_WIDTH, sh=VIDEO_HEIGHT,
            )
            frame = vfx_result["frame"]

            # when the animation finishes, mark it done so game_state resets
            if vfx_result["effect_finished"]:
                self.game.is_effect_complete = True
                self.game.update([])
                self.refresh_strip()

        # --- hud overlay (drawn directly on the frame) ---
        cv2.putText(frame, f"TARGET: {status['target'].upper()}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        hud_text  = "JUTSU ACTIVATED!" if status["is_complete"] else f"NEXT: {status['next_sign'].upper()}"
        hud_color = (0, 255, 180) if status["is_complete"] else (255, 255, 255)
        cv2.putText(frame, hud_text, (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)

        # --- render frame to tkinter canvas ---
        canvas_w = self.video_canvas.winfo_width()  or VIDEO_WIDTH
        canvas_h = self.video_canvas.winfo_height() or VIDEO_HEIGHT

        display = cv2.resize(frame, (canvas_w, canvas_h))
        tk_img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB)))

        if not hasattr(self, "video_image_id"):
            self.video_image_id = self.video_canvas.create_image(0, 0, image=tk_img, anchor=tk.NW)
        else:
            self.video_canvas.itemconfig(self.video_image_id, image=tk_img)

        # keep a reference so python's garbage collector doesn't delete the image
        self.video_canvas.image = tk_img

        self.update_dashboard()
        self.root.after(30, self.update_loop)

    # -------------------------------------------------------------------------
    # dashboard updates
    # -------------------------------------------------------------------------

    def update_dashboard(self):
        """update the text labels and icon highlight colours in the bottom bar."""
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

            for w in self.sign_widgets:
                idx = w["index"]
                name = w["name"]
                normal_img = self.sign_images.get(f"{name}_{idx}_normal")
                dimmed_img = self.sign_images.get(f"{name}_{idx}_dimmed", normal_img)

                if idx < current_idx:
                    # already completed — dimmed + green border
                    w["lbl_img"].config(image=dimmed_img)
                    w["lbl_img"].image = dimmed_img
                    w["border_frame"].config(highlightbackground="#00ff00")
                    w["lbl_text"].config(fg="#888")
                elif idx == current_idx and not is_complete:
                    # current target — bright + orange border
                    w["lbl_img"].config(image=normal_img)
                    w["lbl_img"].image = normal_img
                    w["border_frame"].config(highlightbackground="#ffaa00")
                    w["lbl_text"].config(fg="#ffaa00")
                else:
                    # upcoming — normal + no border
                    w["lbl_img"].config(image=normal_img)
                    w["lbl_img"].image = normal_img
                    w["border_frame"].config(highlightbackground="#1a1a1a")
                    w["lbl_text"].config(fg="white")

        except Exception as e:
            print(f"dashboard error: {e}")

    # -------------------------------------------------------------------------
    # button callbacks
    # -------------------------------------------------------------------------

    def cycle_jutsu(self):
        """skip to the next jutsu: clear effects, advance game state, refresh ui."""
        clear_all_vfx()
        self.game.next_jutsu()
        self.refresh_strip()
        self.update_dashboard()

    def on_closing(self):
        """release camera and close window cleanly."""
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = NarutoApp(root)
    root.mainloop()