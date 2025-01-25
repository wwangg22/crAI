import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agents import *
from agents import CRPPOAgent
from other_helpers import Elixir
from tower_health import TowerHealth
from yolo_helper import YoloModel
from mss import mss
import cv2
from contact import SendCommand
import time


class CRAgent():

    def __init__(self, observation_dim, discrete = True):

        action_dim = [self.getActionDim(), self.getCardDim(), self.getBoardDim()]
        self.actor = CRPPOAgent(observation_dim, [self.getCardDim()+1,  self.getBoardDim()], discrete = discrete)
    
    def getBoardMask(self, type):
        if type == "troop_n":
            mask = np.ones(self.getBoardDim())

            #corner of the board
            mask[:6] = 0
            mask[12:18] = 0
            mask[31*18:31*18+6] = 0
            mask[31*18+12:31*18+18] = 0

            #river in the middle
            mask[14*18:14*18+1] = 0
            mask[14*18+17:14*18+18] = 0
            mask[17*18:17*18+1] = 0
            mask[17*18+17:17*18+18] = 0

            mask[15*18:17*18]=0

            #left princess tower
            mask[24*18+2:24*18+5] = 0
            mask[25*18+2:25*18+5] = 0
            mask[26*18+2:26*18+5] = 0

            #right princess tower
            mask[24*18+13:24*18+16] = 0
            mask[25*18+13:25*18+16] = 0
            mask[26*18+13:26*18+16] = 0

            #center king tower
            mask[30*18+7:30*18+11]=0
            mask[29*18+7:29*18+11]=0
            mask[28*18+7:28*18+11]=0
            mask[27*18+7:27*18+11]=0

            #enemy half 0
            mask[:15*18] = 0

            return mask
        elif type == "troop_r":
            mask = np.ones(self.getBoardDim())

            #corner of the board
            mask[:6] = 0
            mask[12:18] = 0
            mask[31*18:31*18+6] = 0
            mask[31*18+12:31*18+18] = 0

            #river in the middle
            mask[14*18:14*18+1] = 0
            mask[14*18+17:14*18+18] = 0
            mask[17*18:17*18+1] = 0
            mask[17*18+17:17*18+18] = 0

            mask[15*18:17*18]=0

            #left princess tower
            mask[24*18+2:24*18+5] = 0
            mask[25*18+2:25*18+5] = 0
            mask[26*18+2:26*18+5] = 0

            #right princess tower
            mask[24*18+13:24*18+16] = 0
            mask[25*18+13:25*18+16] = 0
            mask[26*18+13:26*18+16] = 0

            #center king tower
            mask[30*18+7:30*18+11]=0
            mask[29*18+7:29*18+11]=0
            mask[28*18+7:28*18+11]=0
            mask[27*18+7:27*18+11]=0

            #mask out top
            mask[0:11*18] = 0
            
            #mask out left side
            mask[11*18:11*18+9] = 0
            mask[12*18:12*18+9] = 0
            mask[13*18:13*18+9] = 0
            mask[14*18:14*18+9] = 0
            return mask
        elif type == "troop_l":
            mask = np.ones(self.getBoardDim())

            #corner of the board
            mask[:6] = 0
            mask[12:18] = 0
            mask[31*18:31*18+6] = 0
            mask[31*18+12:31*18+18] = 0

            #river in the middle
            mask[14*18:14*18+1] = 0
            mask[14*18+17:14*18+18] = 0
            mask[17*18:17*18+1] = 0
            mask[17*18+17:17*18+18] = 0

            mask[15*18:17*18]=0

            #left princess tower
            mask[24*18+2:24*18+5] = 0
            mask[25*18+2:25*18+5] = 0
            mask[26*18+2:26*18+5] = 0

            #right princess tower
            mask[24*18+13:24*18+16] = 0
            mask[25*18+13:25*18+16] = 0
            mask[26*18+13:26*18+16] = 0

            #center king tower
            mask[30*18+7:30*18+11]=0
            mask[29*18+7:29*18+11]=0
            mask[28*18+7:28*18+11]=0
            mask[27*18+7:27*18+11]=0

            #mask out top
            mask[0:11*18] = 0
            
            #mask out right side
            mask[11*18+9:11*18+18] = 0
            mask[12*18+9:12*18+18] = 0
            mask[13*18+9:13*18+18] = 0
            mask[14*18+9:14*18+18] = 0
            return mask
        elif type == "troop_b":
            mask = np.ones(self.getBoardDim())

            #corner of the board
            mask[:6] = 0
            mask[12:18] = 0
            mask[31*18:31*18+6] = 0
            mask[31*18+12:31*18+18] = 0

            #river in the middle
            mask[14*18:14*18+1] = 0
            mask[14*18+17:14*18+18] = 0
            mask[17*18:17*18+1] = 0
            mask[17*18+17:17*18+18] = 0

            mask[15*18:17*18]=0

            #left princess tower
            mask[24*18+2:24*18+5] = 0
            mask[25*18+2:25*18+5] = 0
            mask[26*18+2:26*18+5] = 0

            #right princess tower
            mask[24*18+13:24*18+16] = 0
            mask[25*18+13:25*18+16] = 0
            mask[26*18+13:26*18+16] = 0

            #center king tower
            mask[30*18+7:30*18+11]=0
            mask[29*18+7:29*18+11]=0
            mask[28*18+7:28*18+11]=0
            mask[27*18+7:27*18+11]=0

            #mask out top
            mask[0:11*18] = 0

            return mask
        elif type == "troop_noaction":
            mask = np.zeros(self.getBoardDim())

            mask[0] = 1
            return mask
        elif type == "spell":
            mask = np.ones(self.getBoardDim())

            #corner of the board
            mask[:6] = 0
            mask[12:18] = 0
            mask[31*18:31*18+6] = 0
            mask[31*18+12:31*18+18] = 0

            return mask
    def getDeckMask(self, arr):
        #arr is an array of all the cards that are in the deck and can or cannot be placed
        mask = np.ones(self.getCardDim()+1)
        for i in range(len(arr)):
            mask[i+1] = 0
    def getBoardDim(self):
        return 18*32
    
    def getActionDim(self):
        return 2
    
    def getCardDim(self):
        return 4
    
    def getAction(self, observation, mask=None):
        return self.actor.get_action(observation, mask)


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
    contact = SendCommand(29)

    tower_viewer = TowerHealth(level=9)
    agent = CRAgent(troop_viewer.getDim() + deck_viewer.getDim() + tower_viewer.getDim()+2)
    # print(tower_viewer.getDim() + troop_viewer.getDim() + deck_viewer.getDim()+2)
    contact.connect()
    while True:

        tick=1# tick = int(contact.get_tick()) / 720
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        pure_frame = frame_bgr.copy()
        tower_viewer.update(pure_frame, tower_monitor=health_bars)
        elixir_viewer.detect_elixir(pure_frame)
        elixir = elixir_viewer.getElixir()
        troops = troop_viewer.predict(frame_bgr)
        deck = deck_viewer.predict(frame_bgr)

        obs = np.concatenate([troops, deck, tower_viewer.getAllHealth(), elixir , [tick]])
        print(len(obs))
        deck_mask = []
        deck_mask.append(np.array([1, deck[0], deck[4], deck[8], deck[12]]))
        deck_mask.append(agent.getBoardMask("troop_noaction"))
        for i in range(4):
            if deck[i*4 + 3] == 1 and deck[i*4 + 2] == 0 and deck[i*4 + 1] == 0:
                deck_mask.append(agent.getBoardMask("spell"))
            else:
                if tower_viewer.getAllHealth()[0] == 0:
                    if tower_viewer.getAllHealth()[1] == 0:
                        deck_mask.append(agent.getBoardMask("troop_b"))
                    else:
                        deck_mask.append(agent.getBoardkMask("troop_l"))
                elif tower_viewer.getAllHealth()[1] == 0:
                    deck_mask.append(agent.getBoardMask("troop_r"))
                else:
                    deck_mask.append(agent.getBoardMask("troop_n"))
    
        action, log_prob, mask1, mask2 = agent.getAction(obs, mask=deck_mask)

        if action[0] > 0:
            troop_num = 4*deck[(action[0]-1) * 4 +1] + 2 *deck[(action[0]-1) * 4 +2] + deck[(action[0]-1) * 4 +3]
            x = action[1] % 18
            y = action[1] // 18
            x = 1000*x + 500
            y = 1000*y + 500
            contact.place_troop(troop_num, x, y)
        time.sleep(0.5)
