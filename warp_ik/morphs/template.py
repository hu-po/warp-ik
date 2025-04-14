import time

from warp_ik.src.morph import BaseMorph

class Morph(BaseMorph):

    def _update_config(self):
        self.config.train_iters = 2 # min training iterations
        self.config.config_extras = {
            "sleep_time": 0.01,
        }

    def _step(self):
        time.sleep(self.config.config_extras["sleep_time"])