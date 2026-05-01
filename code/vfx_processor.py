import os
import cv2
import numpy as np

ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')

# maps each jutsu name to its vfx type and asset paths.
# "hand"     = effect follows the player's hand position
# "sequence" = animation plays first, then a still image appears
# "combo"    = smoke animation + a character reveal image
VFX_CONFIG = {
    "Rasengan":           {"type": "hand",     "path": "rasengan_frames", "secondary": None},
    "Chidori":            {"type": "hand",     "path": "chidori_frames",  "secondary": None},
    "Fireball Jutsu":     {"type": "sequence", "path": "fire_frames",     "secondary": "fireball.png"},
    "Shadow Clone Jutsu": {"type": "combo",    "path": "smoke_frames",    "secondary": "naruto_clone.png"},
    "Summoning Jutsu":    {"type": "combo",    "path": "smoke_frames",    "secondary": "kurama.webp"},
}


class JutsuVFX:
    """
    holds all the frames for one jutsu's vfx and tracks playback state.
    instantiated once per active jutsu, then discarded when it finishes.
    """

    def __init__(self, name, config):
        self.name = name
        self.type = config["type"]
        self.current_frame = 0

        # load every png frame from the folder into memory as a list
        frames_folder = os.path.join(ASSETS_DIR, config["path"])
        self.frames = self._load_frames(frames_folder)

        # load the optional secondary image (e.g. naruto_clone.png, fireball.png)
        self.secondary_asset = None
        if config["secondary"]:
            path = os.path.join(ASSETS_DIR, config["secondary"])
            self.secondary_asset = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # track playback progress
        self.primary_finished = False
        self.secondary_count = 0
        self.secondary_max_count = 78  # how many frames to show the secondary image
        self.secondary_finished = (self.secondary_asset is None)

    def _load_frames(self, folder_path):
        """load all image files from a folder, sorted by filename."""
        if not os.path.exists(folder_path):
            print(f"[ERROR] VFX frames folder not found: {folder_path}")
            return []
        files = sorted(f for f in os.listdir(folder_path) if f.endswith(".png"))
        print(f"[DEBUG] VFX: Loading {len(files)} frames from {folder_path}")
        frames = [cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_UNCHANGED) for f in files]
        print(f"[DEBUG] VFX: Successfully loaded {len([f for f in frames if f is not None])} frames")
        return frames

    def get_next_frame(self):
        """return the next animation frame. returns last frame when animation ends."""
        if not self.frames:
            return None
        if self.primary_finished:
            return self.frames[-1]

        frame = self.frames[self.current_frame]
        self.current_frame += 1

        if self.current_frame >= len(self.frames):
            self.primary_finished = True

        return frame

    def tick_secondary(self):
        """count down how long the secondary image stays on screen."""
        self.secondary_count += 1
        if self.secondary_count >= self.secondary_max_count:
            self.secondary_finished = True


# --- global state: one entry per currently-playing jutsu ---
active_vfx = {}


def clear_all_vfx():
    """stop all playing effects immediately (called when skipping jutsu)."""
    active_vfx.clear()


def overlay_effect(bg, jutsu_name, x=None, y=None, size=350, sw=900, sh=450):
    """
    main entry point. called every frame when a jutsu is complete.
    draws the vfx animation on top of the camera feed (bg).

    returns a dict:
        'frame'           — the modified frame with vfx drawn on it
        'effect_finished' — True when the full animation is done
    """
    # create a new vfx instance the first time this jutsu is called
    if jutsu_name not in active_vfx:
        if jutsu_name not in VFX_CONFIG:
            print(f"[ERROR] VFX: no config found for jutsu '{jutsu_name}'. Available: {list(VFX_CONFIG.keys())}")
            return {"frame": bg, "effect_finished": False}
        print(f"[DEBUG] VFX: Creating VFX for '{jutsu_name}'")
        active_vfx[jutsu_name] = JutsuVFX(jutsu_name, VFX_CONFIG[jutsu_name])
        print(f"[DEBUG] VFX: '{jutsu_name}' loaded with {len(active_vfx[jutsu_name].frames)} frames")

    vfx = active_vfx[jutsu_name]
    effect_img = vfx.get_next_frame()

    if effect_img is None:
        return {"frame": bg, "effect_finished": False}

    result_frame = bg
    is_finished = False

    # default position is screen center if no hand is detected
    cx = x if x is not None else sw // 2
    cy = y if y is not None else sh // 2

    if vfx.type == "hand":
        # effect follows hand position; done when animation ends
        if vfx.primary_finished:
            is_finished = True
        else:
            result_frame = _blend_overlay(bg, effect_img, cx, cy, size)

    elif vfx.type == "combo":
        # smoke plays over full screen, then character image fades in
        if vfx.primary_finished and vfx.secondary_finished:
            is_finished = True
        else:
            if not vfx.secondary_finished and (vfx.current_frame >= 5 or vfx.primary_finished):
                result_frame = _blend_overlay(bg, vfx.secondary_asset, sw // 2, sh // 2, sw=sw, sh=sh, fullscreen=True)
                vfx.tick_secondary()
            if not vfx.primary_finished:
                result_frame = _blend_overlay(result_frame, effect_img, sw // 2, sh // 2, sw=sw, sh=sh, fullscreen=True)

    elif vfx.type == "sequence":
        # animation plays first, then a still image shows at hand position
        if vfx.secondary_finished:
            is_finished = True
        else:
            if not vfx.primary_finished:
                result_frame = _blend_overlay(bg, effect_img, sw // 2, sh // 2, sw=sw, sh=sh, fullscreen=True)
            elif vfx.secondary_asset is not None:
                vfx.tick_secondary()
                result_frame = _blend_overlay(bg, vfx.secondary_asset, cx, cy, size)

    if is_finished:
        del active_vfx[jutsu_name]
        return {"frame": bg, "effect_finished": True}

    return {"frame": result_frame, "effect_finished": False}


def _blend_overlay(bg, fg, x, y, size=None, sw=900, sh=450, fullscreen=False):
    """
    alpha-blend a foreground image (fg) onto a background frame (bg).
    uses numpy vectorization for speed — no pixel-by-pixel loops.
    """
    try:
        if fullscreen:
            fg = cv2.resize(fg, (sw, sh))
            x1, y1 = 0, 0
            h, w = sh, sw
        else:
            if size:
                fg = cv2.resize(fg, (size, size))
            h, w = fg.shape[:2]
            x1 = x - w // 2
            y1 = y - h // 2

        x2, y2 = x1 + w, y1 + h
        bg_h, bg_w = bg.shape[:2]

        # clamp to frame boundaries so we never go out of bounds
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(bg_w, x2), min(bg_h, y2)

        if cx1 >= cx2 or cy1 >= cy2:
            return bg  # completely off screen, nothing to draw

        # crop the fg to only the visible portion
        fg_crop = fg[max(0, -y1): max(0, -y1) + (cy2 - cy1),
                     max(0, -x1): max(0, -x1) + (cx2 - cx1)]
        roi = bg[cy1:cy2, cx1:cx2]

        # alpha channel 0–255 → 0.0–1.0 for blending math
        alpha = fg_crop[:, :, 3:4] / 255.0
        fg_rgb = fg_crop[:, :, :3]

        # blend: result = alpha * fg + (1 - alpha) * bg
        blended = (alpha * fg_rgb + (1 - alpha) * roi).astype(np.uint8)
        bg[cy1:cy2, cx1:cx2] = blended

        return bg

    except Exception as e:
        print(f"vfx blend error: {e}")
        return bg