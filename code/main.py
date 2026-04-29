import cv2
import tkinter as tk
from PIL import Image, ImageTk, ImageFilter, ImageEnhance

from detector import JutsuDetector

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
        self.current_win_w = 900
        self.current_win_h = 600

        self.setup_ui()

        # CAMERA SETUP
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

        # STATE
        #self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # LOOP
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
            
            # 1. AI DETECTION
            # Detect jutsus BEFORE resize so the AI sees the original quality
            detections = self.detector.detect(frame)

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
