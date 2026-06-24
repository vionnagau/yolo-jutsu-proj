import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import numpy as np
import os
from PIL import Image

# Force paths relative to this root file to keep asset lookups clean
import sys
sys.path.append(os.path.abspath("./code"))

# Import your existing engine classes safely
from detector import JutsuDetector
from game_state import JutsuGame

st.set_page_config(page_title="NinjaSight AR Demo", layout="wide")

st.title("🥷 NinjaSight: Real-Time Hand Sign AR System")
st.markdown("Show consecutive Naruto hand signs to your webcam to compile and execute a Ninjutsu sequence!")

# Cache the AI Model initialization so it does not reload on every single frame refresh
@st.cache_resource
def load_detection_models():
    # Looks for model files in the project root path structure
    detector = JutsuDetector(model_path="model/best.pt")
    return detector

try:
    detector_engine = load_detection_models()
    # Initialize game state instance in the global session context
    if "game_engine" not in st.session_state:
        st.session_state.game_engine = JutsuGame()
except Exception as e:
    st.error(f"Error initializing underlying AI models: {e}")

class JutsuVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = detector_engine

    def recv(self, frame):
        # Transform stream frame to native numpy BGR array for OpenCV compatibility
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirroring frame for user UX consistency
        
        # Access game machine state tracking safely across video loop
        game = st.session_state.game_engine
        status = game.get_status()
        
        # --- PHASE 1: Sign Detection ---
        if not status["is_complete"]:
            detections = self.detector.detect(img)
            # Normalize sign inputs to lowercase matches
            detected_labels = [d[0].lower().strip() for d in detections]
            
            # Feed input directly into your pre-existing validation engine
            game.update(detected_labels)
            
            # Draw visual tracking indicator bounding boxes onto the active frame stream
            for label, conf, (x1, y1, x2, y2) in detections:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label.upper()} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # --- PHASE 2: UI Status Text Burn-in ---
        # Draw status metrics straight onto the video layer since Tkinter elements are absent
        current_status = game.get_status()
        target_jutsu = current_status["target"].upper()
        next_sign = current_status["next_sign"].upper()
        
        cv2.putText(img, f"TARGET JUTSU: {target_jutsu}", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        if current_status["is_complete"]:
            cv2.putText(img, "JUTSU ACTIVATED!", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(img, f"NEXT SIGN REQUIRED: {next_sign}", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame.from_ndarray(img, format="bgr24")

# Establish web rendering component container layout blocks
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Webcam Stream Input")
    # Deploy secure WebRTC connection engine interface for processing web frames
    ctx = webrtc_streamer(
        key="jutsu-recognition-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=JutsuVideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("Mission Status Machine")
    if "game_engine" in st.session_state:
        st.button("🔄 Reset Current Sequence Tracking", on_click=lambda: st.session_state.game_engine.update([]))
        
        # Pull latest evaluation telemetry data frame to update static panel data fields
        current_telemetry = st.session_state.game_engine.get_status()
        st.metric(label="Target Jutsu Goal", value=current_telemetry["target"].title())
        st.metric(label="Awaiting Gesture Sign", value=current_telemetry["next_sign"].upper())
        
        if current_telemetry["is_complete"]:
            st.balloons()
            st.success("✨ Sequence Success! Jutsu Completed.")