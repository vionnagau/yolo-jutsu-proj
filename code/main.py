import cv2
import tkinter as tk
from PIL import Image, ImageTk, ImageFilter, ImageEnhance

from detector import JutsuDetector

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

        self.setup_ui()

        # CAMERA SETUP
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

        # STATE
        self.running = True

        # LOOP
        self.update_loop()
        
    def setup_ui(self):
        self.video_canvas = tk.Canvas(
            self.root,
            width=VIDEO_WIDTH,
            height=VIDEO_HEIGHT,
            bg="black",
            highlightthickness=0
        )
        self.video_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
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

            # Tkinter and PIL follow "RGB" colour order
            # Whereas the OpenCV library follows "BGR" colour order
            # So we need to convert it to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tk_img = ImageTk.PhotoImage(image=pil_img)

            # First frame: We create the image on the canvas
            # Every frame after that: We're just updating it constantly
            if not hasattr(self, 'video_image_id'):
                self.video_image_id = self.video_canvas.create_image(
                    0, 0, image = tk_img, anchor = tk.NW
                )
            else:
                self.video_canvas.itemconfig(self.video_image_id, image = tk_img)

            # We need to keep a reference of the image otherwise Python will just garbage-collect the image
            self.video_canvas.image = tk_img

        # We're calling this function to update the frame every 30ms
        self.root.after(30, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = NarutoApp(root)
    root.mainloop()
