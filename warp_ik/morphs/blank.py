import time

from src.morph import BaseMorph

class Morph(BaseMorph):

    def _step(self):
        time.sleep(0.01)