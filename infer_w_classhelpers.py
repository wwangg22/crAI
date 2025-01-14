from other_helpers import Elixir
from tower_health import TowerHealth
from yolo_helper import YoloModel
from mss import mss
import torch
import numpy as np
import cv2
import time
from contact import SendCommand

monitor = {
    "top": 50,      # Y-position of the top edge (absolute screen coordinate)
    "left": 0,      # X-position of the left edge (absolute screen coordinate)
    "width": 720,   # Capture width
    "height": 1300  # Capture height
}

health_bars = [
    {
        "name": "cfk_health",
        "top": 950,
        "left": 275,
        "width": 200,
        "height": 100,
    },
    {
        "name": "cek_health",
        "top": 0,
        "left": 275,
        "width": 200,
        "height": 100,
    },
    {
        "name": "let_health",
        "top": 125,
        "left": 75,
        "width": 200,
        "height": 100,
    },
    {
        "name": "ret_health",
        "top": 125,
        "left": 450,
        "width": 200,
        "height": 100,
    },
    {
        "name": "lft_health",
        "top": 750,
        "left": 75,
        "width": 200,
        "height": 100,
    },
    {
        "name": "rft_health",
        "top": 750,
        "left": 450,
        "width": 200,
        "height": 100,
    },
    
]

sct = mss()

if __name__=="__main__":
    troop_viewer = YoloModel(model_path="./weights/game/best.pt", img_size = 600, deck = False) #./yolov5/runs/train/yolov5s_w_sprites/weights/best.pt
    deck_viewer = YoloModel(model_path="./weights/deck/best.pt", img_size = 200, deck=True) #./yolov5/runs/train/finetune_new2/weights/best.pt

    elixir_viewer = Elixir()
    contact = SendCommand(26)

    tower_viewer = TowerHealth(level=9)
    contact.connect()
    while True:
        contact.get_tick()
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        pure_frame = frame_bgr.copy()

        troops = troop_viewer.predict(frame_bgr)
        deck = deck_viewer.predict(frame_bgr)
        tower_viewer.update(pure_frame, tower_monitor=health_bars)

        print(tower_viewer.getAllHealth())






