import numpy as np
class Elixir():

    def __init__(self, offset = 0):
        self.elixir = None
        self.elixir_points = [
            i * 53 + 222 +offset for i in reversed(range(10))
        ]
        self.elixir_color = [
            208, 32, 216
        ]

    def detect_elixir(self, frame, y = 245, tolerance = 80):
        elixir = 10
        for idx, x in enumerate(self.elixir_points):
            met = True
            px_color = frame[y,x]
            # print(px_color)
            for i in range(3):  # B, G, R channels
                if not (self.elixir_color[i] - tolerance <= px_color[i] <= self.elixir_color[i] + tolerance):       
                    elixir -= 1
                    met = False
                    break
            if met:  
                break
        self.elixir = elixir

    def getElixir(self):
        return np.full((1,), self.elixir/10)
    
    def getDim(self):
        return 1