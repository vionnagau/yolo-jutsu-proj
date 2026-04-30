import time

class JutsuGame:
    def __init__(self):
        self.JUTSUS = {
            "Chidori": ["Bird", "Serpent", "Monkey"],
            "Fireball Jutsu": ["Serpent", "Ram", "Monkey", "Boar", "Horse", "Tiger"],
            "Shadow Clone Jutsu": ["Tiger"],
            "Summoning Jutsu": ["Boar", "Dog", "Bird", "Monkey", "Ram"],
            "Rasengan": ["Tiger", "Boar"]
        }

        self.jutsu_order = list(self.JUTSUS.keys())
        self.current_jutsu_index = 0

        self.target_jutsu = self.jutsu_order[self.current_jutsu_index]
        self.sequence = self.JUTSUS[self.target_jutsu]
        self.current_step_index = 0

        self.hold_start_time = 0
        self.required_hold_duration = 0.2  # seconds
        self.current_detected_sign = None

        self.is_complete = False
        self.is_effect_complete = False
        self.complete_time = 0

    def reset(self):
        self.current_step_index = 0
        self.is_complete = False
        self.is_effect_complete = False
        self.current_detected_sign = None
        self.hold_start_time = 0
        
    def next_jutsu(self):
        self.current_jutsu_index = (self.current_jutsu_index + 1) % len(self.jutsu_order)
        self.target_jutsu = self.jutsu_order[self.current_jutsu_index]
        self.sequence = self.JUTSUS[self.target_jutsu]
        self.reset()

    def update(self, detected_labels):
        now = time.time()

        if self.is_complete:
            if self.is_effect_complete:
                self.reset()
            return

        if self.current_step_index < len(self.sequence):
            expected_sign = self.sequence[self.current_step_index].lower()
            found = any(label.lower() == expected_sign for label in detected_labels)

            if found:
                if self.current_detected_sign == expected_sign:
                    if now - self.hold_start_time > self.required_hold_duration:
                        self.current_step_index += 1
                        self.current_detected_sign = None
                        self.hold_start_time = 0

                        if self.current_step_index >= len(self.sequence):
                            self.is_complete = True
                            self.complete_time = now
                    else:
                        self.current_detected_sign = expected_sign
                        self.hold_start_time = now
            else:
                self.current_detected_sign = None
                self.hold_start_time = 0
                
    def get_status(self):
        return {
            "target": self.target_jutsu,
            "sequence": self.sequence,
            "current_index": self.current_step_index,
            "is_complete": self.is_complete,
            "is_effect_complete": self.is_effect_complete,
            "next_sign": "DONE" if self.is_complete else self.sequence[self.current_step_index]
        }

