import os
import cv2
import numpy as np

ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')

VFX_CONFIG = {
    "Rasengan": {"type": "hand", "path": "rasengan_frames", "secondary": None},
    "Chidori": {"type": "hand", "path": "chidori_frames", "secondary": None},
    "Fireball Jutsu": {"type": "sequence", "path": "fire_frames", "secondary": "fireball.png"},
    "Shadow Clone Jutsu": {"type": "combo", "path": "smoke_frames", "secondary": "naruto_clone.png"},
    "Summoning Jutsu": {"type": "combo", "path": "smoke_frames", "secondary": "kurama.webp"}
}

class JutsuVFX:
    def __init__(self, name, config):
        self.name = name
        self.type = config['type']
        self.current_frame = 0
        self.frames = self._load_frames(os.path.join(ASSETS_DIR, config['path']))
        
        self.secondary_asset = None
        if config['secondary']:
            path = os.path.join(ASSETS_DIR, config['secondary'])
            self.secondary_asset = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        self.secondary_count = 0
        self.secondary_max_count = 78
        self.primary_finished = False
        self.secondary_finished = True if self.secondary_asset is None else False
        
    def _load_frames(self, path):
        if not os.path.exists(path): return []
        files = sorted([f for f in os.listdir(path)])
        return [cv2.imread(os.path.join(path, f), cv2.IMREAD_UNCHANGED) for f in files]

    def get_frame(self):
        if not self.frames: return None
        if self.primary_finished: return self.frames[-1]

        frame = self.frames[self.current_frame]
        self.current_frame += 1
        
        if self.current_frame >= len(self.frames):
            self.primary_finished = True
        return frame

    def update_secondary(self):
        self.secondary_count += 1
        if self.secondary_count >= self.secondary_max_count:
            self.secondary_finished = True
            
# GLOBAL STATE
active_vfx = {}

def clear_all_vfx():
    active_vfx.clear()

def overlay_effect(bg, jutsu_name, x=None, y=None, size=350, sw=900, sh=450):
    if jutsu_name not in active_vfx:
        if jutsu_name not in VFX_CONFIG:
            return {'frame': bg, 'effect_finished': False}
        active_vfx[jutsu_name] = JutsuVFX(jutsu_name, VFX_CONFIG[jutsu_name])

    vfx = active_vfx[jutsu_name]
    effect_img = vfx.get_frame()
    
    if effect_img is None: 
        return {'frame': bg, 'effect_finished': False}

    res_frame = bg
    is_finished = False
    center_x, center_y = (x if x else sw//2), (y if y else sh//2)

    # Rendering Logic
    if vfx.type == "hand":
        if vfx.primary_finished: is_finished = True
        else: res_frame = _apply_alpha_overlay(bg, effect_img, center_x, center_y, size)

    elif vfx.type == "combo":
        if vfx.primary_finished and vfx.secondary_finished:
            is_finished = True
        else:
            if (vfx.current_frame >= 5 or vfx.primary_finished) and not vfx.secondary_finished:
                if vfx.secondary_asset is not None:
                    res_frame = _apply_alpha_overlay(bg, vfx.secondary_asset, sw//2, sh//2, sw=sw, sh=sh, is_full=True)
                    vfx.update_secondary()
            if not vfx.primary_finished: 
                res_frame = _apply_alpha_overlay(res_frame, effect_img, sw//2, sh//2, sw=sw, sh=sh, is_full=True)

    elif vfx.type == "sequence":
        if vfx.secondary_finished:
            is_finished = True
        else:
            if not vfx.primary_finished:
                res_frame = _apply_alpha_overlay(bg, effect_img, sw//2, sh//2, sw=sw, sh=sh, is_full=True)
            elif vfx.secondary_asset is not None:
                vfx.update_secondary()
                res_frame = _apply_alpha_overlay(bg, vfx.secondary_asset, center_x, center_y, size)

    if is_finished:
        del active_vfx[jutsu_name]
        return {'frame': bg, 'effect_finished': True}

    return {'frame': res_frame, 'effect_finished': False}

def _apply_alpha_overlay(bg, fg, x, y, size=None, sw=900, sh=450, is_full=False):
    # Optimized alpha blending using NumPy vectorization
    try:
        if is_full:
            fg = cv2.resize(fg, (sw, sh))
            h, w, y1, x1 = sh, sw, 0, 0
        else:
            if size: fg = cv2.resize(fg, (size, size))
            h, w = fg.shape[:2]
            y1, x1 = y - h // 2, x - w // 2
        
        y2, x2 = y1 + h, x1 + w
        bg_h, bg_w = bg.shape[:2]

        y1_c, y2_c = max(0, y1), min(bg_h, y2)
        x1_c, x2_c = max(0, x1), min(bg_w, x2)
        
        if y1_c >= y2_c or x1_c >= x2_c: return bg

        fg_y1, fg_x1 = max(0, -y1), max(0, -x1)
        fg_part = fg[fg_y1:fg_y1 + (y2_c - y1_c), fg_x1:fg_x1 + (x2_c - x1_c)]
        roi = bg[y1_c:y2_c, x1_c:x2_c]

        # Vectorized Alpha Blending - Ensure explicit broadcasting
        alpha = fg_part[:, :, 3:4] / 255.0 
        fg_rgb = fg_part[:, :, :3]
        
        # This formula blends the foreground with the background ROI
        res = (alpha * fg_rgb + (1 - alpha) * roi).astype(np.uint8)
        bg[y1_c:y2_c, x1_c:x2_c] = res # Force the result back into the original image
            
        return bg
    
    except Exception as e:
        print(f"Overlay Error: {e}")
        return bg