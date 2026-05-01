import time


class JutsuGame:
    """
    tracks which jutsu the player is trying to perform,
    checks if the detected hand sign matches the expected one,
    and advances the sequence when the sign is held long enough.
    """

    def __init__(self):
        # all available jutsus and their required hand sign sequences
        self.JUTSUS = {
            "Chidori":            ["Bird", "Serpent", "Monkey"],
            "Fireball Jutsu":     ["Serpent", "Ram", "Monkey", "Boar", "Horse", "Tiger"],
            "Shadow Clone Jutsu": ["Tiger"],
            "Summoning Jutsu":    ["Boar", "Dog", "Bird", "Monkey", "Ram"],
            "Rasengan":           ["Tiger", "Boar"],
        }

        self.jutsu_order = list(self.JUTSUS.keys())
        self.current_jutsu_index = 0

        # load first jutsu
        self.target_jutsu = self.jutsu_order[self.current_jutsu_index]
        self.sequence = self.JUTSUS[self.target_jutsu]

        # how long (seconds) player must hold a sign before it counts
        self.required_hold_duration = 0.5

        # reset all tracking state
        self.reset()

    def reset(self):
        """clear progress for the current jutsu so it can be attempted again."""
        self.current_step_index = 0
        self.is_complete = False
        self.is_effect_complete = False
        self.current_detected_sign = None
        self.hold_start_time = 0

    def next_jutsu(self):
        """cycle to the next jutsu in the list (wraps around)."""
        self.current_jutsu_index = (self.current_jutsu_index + 1) % len(self.jutsu_order)
        self.target_jutsu = self.jutsu_order[self.current_jutsu_index]
        self.sequence = self.JUTSUS[self.target_jutsu]
        self.reset()

    def update(self, detected_labels):
        """
        called every frame with a list of label strings from yolo.
        advances the sequence if the correct sign is held long enough.
        resets to the beginning if the vfx effect has finished playing.
        """
        now = time.time()

        # if the jutsu is done, wait for the vfx to finish, then reset
        if self.is_complete:
            print(f"[DEBUG] Game: jutsu complete, waiting for effect to finish...")
            if self.is_effect_complete:
                print(f"[DEBUG] Game: effect finished, resetting for new round")
                self.reset()
            return

        expected_sign = self.sequence[self.current_step_index].lower()
        print(f"[DEBUG] Game: Expected '{expected_sign}', Detected labels: {detected_labels}")

        # check if any detected label matches what we expect right now
        sign_found = any(label.lower() == expected_sign for label in detected_labels)
        print(f"[DEBUG] Game: Sign found? {sign_found}")

        if sign_found:
            if self.current_detected_sign != expected_sign:
                # --- first frame this sign appears: start the hold timer ---
                print(f"[DEBUG] Game: First detection of '{expected_sign}', starting hold timer")
                self.current_detected_sign = expected_sign
                self.hold_start_time = now
            else:
                # --- sign is being held: check if held long enough ---
                held_duration = now - self.hold_start_time
                print(f"[DEBUG] Game: Holding '{expected_sign}' for {held_duration:.2f}s (need {self.required_hold_duration}s)")
                if held_duration >= self.required_hold_duration:
                    # advance to next step in the sequence
                    print(f"[DEBUG] Game: Hold duration met! Advancing to step {self.current_step_index + 1}")
                    self.current_step_index += 1
                    self.current_detected_sign = None
                    self.hold_start_time = 0

                    # check if the full sequence is complete
                    if self.current_step_index >= len(self.sequence):
                        print(f"[DEBUG] Game: SEQUENCE COMPLETE! Setting is_complete=True")
                        self.is_complete = True
        else:
            # sign was lost — reset hold tracking
            if self.current_detected_sign is not None:
                print(f"[DEBUG] Game: Sign lost, resetting hold timer")
            self.current_detected_sign = None
            self.hold_start_time = 0

    def get_status(self):
        """return a snapshot of the current game state for the ui and vfx."""
        next_sign = "DONE" if self.is_complete else self.sequence[self.current_step_index]
        return {
            "target":          self.target_jutsu,
            "sequence":        self.sequence,
            "current_index":   self.current_step_index,
            "is_complete":     self.is_complete,
            "is_effect_complete": self.is_effect_complete,
            "next_sign":       next_sign,
        }