import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cnn_cragent import CRPPOAgent
from cnn_cragent import MaskMaker
from UTILS.other_helpers import Elixir
from tower_health import TowerHealth
from yolo_helper import YoloModel
from ReplayBuffer import ImageReplayBuffer
from mss import mss
import cv2
from UTILS.contact import SendCommand
import time
import warnings
import pyautogui

# Suppress all warnings (optional)
warnings.filterwarnings("ignore")

####################################################
# ------------------ AGENT WRAPPER -----------------
####################################################
class CRAgent():
    """
    This is your single-agent wrapper which houses references to
    the actor and any local (masking, etc.) logic. It *does not*
    automatically share its actor with another instance, but you
    can force two instances to share the same actor object.
    """
    def __init__(self, observation_dim, discrete=True):
        # Each CRAgent normally has its own CRPPOAgent:
        self.actor = CRPPOAgent(observation_dim, 
                                [self.getCardDim()+1, self.getBoardDim()],
                                discrete=discrete)
        self.maskmaster = MaskMaker()  # placeholder, if you use one

    def getBoardMask(self, type):
        if type == "troop_n":
            return 0
        elif type == "troop_r":
            return 1
        elif type == "troop_l":
            return 2
        elif type == "troop_b":
            return 3
        elif type == "troop_noaction":
            return 4
        elif type == "spell":
            return 5

    def getDeckMask(self, arr):
        # arr is an array of all the cards that are in the deck
        # that can or cannot be placed
        mask = np.ones(self.getCardDim()+1)
        for i in range(len(arr)):
            mask[i+1] = 0

    def getBoardDim(self):
        return 18 * 32
    
    def getActionDim(self):
        return 2
    
    def getCardDim(self):
        return 8

    def getAction(self, image, observation, mask1=None, mask2=None):
        """
        Return (action, log_prob, discrete_mask_1, discrete_mask_2).
        In your code, `mask1`, `mask2` are used to tell the CRPPOAgent
        which actions are valid for the card and board placements.
        """
        return self.actor.get_action(image, observation, mask1, mask2)


####################################################
# ------------ DEFINE CAPTURE REGIONS --------------
####################################################
# First agent sees the game on the left side of screen:
monitor_1 = {
    "top": 50,     
    "left": 0,      
    "width": 720,   
    "height": 1300  
}
card_monitor_1 = {
    "top": 1050,      
    "left": 0,        
    "width": 720,     
    "height": 250     
}
health_bars_1 = [
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

# Second agent sees the same "box" but shifted to the right by left_offset:
left_offset = 760
yolo_monitor = {
    "top": 0,      # Y-position of the top edge (absolute screen coordinate)
    "left": 0,      # X-position of the left edge (absolute screen coordinate)
    "width": 720,   # Capture width
    "height": 1050  # Capture height
}
monitor_2 = {
    "top": 50,
    "left": left_offset,  # SHIFT
    "width": 720,
    "height": 1300
}

####################################################
# ---------------- CREATE HELPERS ------------------
####################################################

sct = mss()
sequence_len = 128
batch_size = 32
buffer_size = 500
UPDATE_INTERVAL = 0.6
LEARN_ITER = 1

# You have one single ReplayBuffer that both agents share:
replay_buffer = ImageReplayBuffer(maxsize=buffer_size, image_shape = (3, 224, 224))

# Create your model *once*, or create them separately and then
# force them to share parameters. Here we create one agent
# for the sake of reference:
 
# Common YOLO models for agent 1:
deck_viewer_1  = YoloModel(model_path="../image_detection_training/weights/deck/exp18best.pt", img_size=200, deck=True)
tower_viewer_1 = TowerHealth(level=9)
elixir_viewer_1 = Elixir()

# Common YOLO models for agent 2 (has its own instance):
deck_viewer_2  = YoloModel(model_path="../image_detection_training/weights/deck/exp18best.pt", img_size=200, deck=True)
tower_viewer_2 = TowerHealth(level=9)
elixir_viewer_2 = Elixir()

# Construct Agent #1:
obs_dim_1 = deck_viewer_1.getDim() + tower_viewer_1.getDim() + 2
agent1 = CRAgent(obs_dim_1)
agent1.actor.load_model()

# Construct Agent #2 with the appropriate observation dimension:
obs_dim_2 = deck_viewer_2.getDim() + tower_viewer_2.getDim() + 2
# agent2 = CRAgent(obs_dim_2)
agent2 = agent1

# Force them to share the same underlying CRPPOAgent (same weights).
# One simple way: you can make agent2.actor point to agent1.actor.
# agent2.actor = agent1.actor

NEW_IP = '206.81.0.61'
# If each agent is supposed to have its *own contact* instance, do it.
contact1 = SendCommand(1, host = NEW_IP)
contact1.connect()
contact1.start_battle()

contact2 = SendCommand(2, host = NEW_IP)  # or a different port, ID, etc.
contact2.connect()
contact2.start_battle()

# If you just have one environment, the code below will need adjusting.
# We'll assume you somehow have two separate "matches" or two separate
# windows, hence contact1 and contact2. 
searching = False

####################################################
# -------------- STORAGE FOR 2 AGENTS --------------
####################################################
# We'll keep separate sequences for each agent, but we feed them all 
# into the SAME replay_buffer.
img_seq_1 = np.zeros((sequence_len + 1, 3, 224, 224))
obs_seq_1       = np.zeros((sequence_len, obs_dim_1))
next_obs_seq_1  = np.zeros((sequence_len, obs_dim_1))
reward_seq_1    = np.zeros(sequence_len)
action_seq_1    = np.zeros((sequence_len, 2))
done_seq_1      = np.zeros(sequence_len)
log_prob_seq_1  = np.zeros(sequence_len)
mask1_seq_1     = torch.zeros((sequence_len, 9))
mask2_seq_1     = np.zeros(sequence_len)

img_seq_2 = np.zeros((sequence_len + 1, 3, 224, 224))
obs_seq_2       = np.zeros((sequence_len, obs_dim_2))
next_obs_seq_2  = np.zeros((sequence_len, obs_dim_2))
reward_seq_2    = np.zeros(sequence_len)
action_seq_2    = np.zeros((sequence_len, 2))
done_seq_2      = np.zeros(sequence_len)
log_prob_seq_2  = np.zeros(sequence_len)
mask1_seq_2     = torch.zeros((sequence_len, 9))
mask2_seq_2     = np.zeros(sequence_len)

ind_1 = 0
ind_2 = 0

last_time = time.time()

####################################################
# --------------------- MAIN LOOP ------------------
####################################################
while True:
    # ---------------------
    # AGENT 1 "ENV" CHECK
    # ---------------------

    # If both are currently out of battle, you might choose to skip 
    # the rest of the loop. Adjust logic as needed:

    if not contact1.inbattle and not contact2.inbattle:
        if not searching:
            searching = True
            contact1.start_battle()
            contact2.start_battle()
            max_duration = 7.0  # seconds, for example

            start_time = time.time()
            while True:
                # If replay buffer has enough data, do a training update
                if replay_buffer.size > batch_size:
                    sample = replay_buffer.sample(batch_size)
                    agent2.actor.learn(sample)  # same as agent1
                else:
                    time.sleep(7)
                    break
                # Check how much time has passed
                elapsed = time.time() - start_time
                if elapsed >= max_duration:
                    break
            contact1.check_inbattle() 
            contact2.check_inbattle()
        time.sleep(1.0)
        continue

    start = time.time()

    # ---------------
    # AGENT 1 STEP
    # ---------------
    if contact1.inbattle and searching:
        # print(contact1.isBattleMaster)
        contact1.request_tick()
        sct_img_1 = sct.grab(monitor_1)
        frame1 = np.array(sct_img_1)
        frame_bgr_1 = cv2.cvtColor(frame1, cv2.COLOR_BGRA2RGB)

        # Tower health
        tower_viewer_1.update(frame_bgr_1, tower_monitor=health_bars_1)

        # Card region for agent1
        card_frame_1 = frame_bgr_1[
            card_monitor_1["top"]:card_monitor_1["top"]+card_monitor_1["height"],
            card_monitor_1["left"]:card_monitor_1["left"]+card_monitor_1["width"]
        ]

        troop_frame_1 = frame_bgr_1[
            card_monitor_1["top"]-100:card_monitor_1["top"],
            card_monitor_1["left"]:card_monitor_1["left"]+card_monitor_1["width"]
        ]
        resized_troop_1 = cv2.resize(troop_frame_1, (224, 224), interpolation=cv2.INTER_LINEAR)
        resized_troop_1 = np.transpose(resized_troop_1, (2, 0, 1))

        # Elixir detection
        elixir_viewer_1.detect_elixir(card_frame_1)
        elixir_1 = elixir_viewer_1.getElixir()

        # YOLO predictions for troops & deck
        deck_1 = deck_viewer_1.predict(card_frame_1)

        reward_1 = tower_viewer_1.getReward()

        # Observations for agent1
        print(elixir_1)
        obs_1 = np.concatenate([
            deck_1,
            tower_viewer_1.getAllHealth(),
            elixir_1,
            [contact1.tick / 720]
        ])
        # print(tower_viewer_1.getAllHealth())

        # Store transitions in sequences
        if ind_1 > 0 and ind_1 % sequence_len == 0:
            # finish the previous chunk
            next_obs_seq_1[ind_1 - 1] = obs_1
            img_seq_1[ind_1] = resized_troop_1

            # advantage calc
            cur_ad_1 = agent1.actor.calculate_advantage(
                img_seq_1[:-1], img_seq_1[1:], obs_seq_1, next_obs_seq_1, reward_seq_1, done_seq_1
            )
            # add to the *shared* replay_buffer
            replay_buffer.add_sequence(
                img_seq_1[:-1], img_seq_1[1:],
                obs_seq_1, next_obs_seq_1, action_seq_1,
                reward_seq_1, done_seq_1, cur_ad_1,
                log_prob_seq_1, mask1_seq_1, mask2_seq_1
            )

            # optional learning step
            if replay_buffer.size > batch_size:
                for _ in range(LEARN_ITER):
                    sample = replay_buffer.sample(batch_size)
                    agent1.actor.learn(sample)  # same as agent2

            # Reset the local buffers
            img_seq_1 = np.zeros((sequence_len + 1, 3, 224, 224))
            obs_seq_1      = np.zeros((sequence_len, obs_dim_1))
            next_obs_seq_1 = np.zeros((sequence_len, obs_dim_1))
            reward_seq_1   = np.zeros(sequence_len)
            action_seq_1   = np.zeros((sequence_len, 2))
            done_seq_1     = np.zeros(sequence_len)
            log_prob_seq_1 = np.zeros(sequence_len)
            mask1_seq_1    = torch.zeros((sequence_len, 9))
            mask2_seq_1    = np.zeros(sequence_len)
            ind_1 = 0

        # Construct masks for agent1 (deck mask, board mask)
        deckmask1_1 = np.zeros(9, dtype=np.int32)
        deckmask1_1[0] = 1
        deckmask1_1[1:]= deck_1[:]
        deckmask2_1 = [None] * 9
        deckmask2_1[0] = agent1.getBoardMask("troop_noaction")
        for ind in range(len(deck_1)):
            if deck_1[ind] == 1:
                if ind == 1:
                    deckmask2_1[ind+1] = agent1.getBoardMask("spell")
                else:
                    if tower_viewer_1.getAllHealth()[0] == 0:
                        if tower_viewer_1.getAllHealth()[1] == 0:
                            deckmask2_1[ind+1] = agent1.getBoardMask("troop_b")
                        else:
                            deckmask2_1[ind+1] = agent1.getBoardMask("troop_l")
                    elif tower_viewer_1.getAllHealth()[1] == 0:
                        deckmask2_1[ind+1] = agent1.getBoardMask("troop_r")
                    else:
                        deckmask2_1[ind+1] = agent1.getBoardMask("troop_n")
                
                
        
        # print(deckmask1_1)
        action_1, log_prob_1, m1_1, m2_1 = agent1.getAction(resized_troop_1, obs_1, mask1=deckmask1_1, mask2=deckmask2_1)

        # If there's a non-zero action in the first dimension, place troop, etc.
        if action_1[0] > 0:
            
            troop_num_1 = action_1[0]-1
            x_1 = action_1[1] % 18
            y_1 = action_1[1] // 18
            if contact1.isBattleMaster:
                x_1 = 1000*x_1 + 500
                y_1 = 1000*y_1 + 500
            else:
                x_1 = 17000 - 1000*x_1 + 500
                y_1 = 31000 - 1000*y_1 + 500
            contact1.place_troop(troop_num_1, x_1, y_1)

        # Save data in local buffer
        img_seq_1[ind_1] = resized_troop_1
        obs_seq_1[ind_1]      = obs_1
        reward_seq_1[ind_1]   = reward_1
        action_seq_1[ind_1]   = action_1
        log_prob_seq_1[ind_1] = log_prob_1
        mask1_seq_1[ind_1]    = m1_1
        mask2_seq_1[ind_1]    = m2_1


        # Next observation for the previous index
        if ind_1 > 0:
            next_obs_seq_1[ind_1 - 1] = obs_1

        contact1.check_inbattle()
        if not contact1.inbattle:
            done_seq_1[ind_1] = 1
            next_obs_seq_1[ind_1] = obs_1
            img_seq_1[ind_1+1] = resized_troop_1

            # Final advantage calc for partial sequence
            cur_ad_1 = agent1.actor.calculate_advantage(
                img_seq_1[:ind_1+1], img_seq_1[1:ind_1+2],
                obs_seq_1[:ind_1+1], next_obs_seq_1[:ind_1+1],
                reward_seq_1[:ind_1+1], done_seq_1[:ind_1+1]
            )
            replay_buffer.add_sequence(
                img_seq_1[:ind_1+1], img_seq_1[1:ind_1+2],
                obs_seq_1[:ind_1+1], next_obs_seq_1[:ind_1+1],
                action_seq_1[:ind_1+1], reward_seq_1[:ind_1+1],
                done_seq_1[:ind_1+1], cur_ad_1[:ind_1+1],
                log_prob_seq_1[:ind_1+1],
                mask1_seq_1[:ind_1+1],
                mask2_seq_1[:ind_1+1]
            )

            # optional learning step
            # if replay_buffer.size > batch_size:
            #     for _ in range(LEARN_ITER):
            #         sample = replay_buffer.sample(batch_size)
            #         agent1.actor.learn(sample)  # same as agent2

            # Reset
            img_seq_1 = np.zeros((sequence_len + 1, 3, 224, 224))
            obs_seq_1      = np.zeros((sequence_len, obs_dim_1))
            next_obs_seq_1 = np.zeros((sequence_len, obs_dim_1))
            reward_seq_1   = np.zeros(sequence_len)
            action_seq_1   = np.zeros((sequence_len, 2))
            done_seq_1     = np.zeros(sequence_len)
            log_prob_seq_1 = np.zeros(sequence_len)
            mask1_seq_1    = torch.zeros((sequence_len, 9))
            mask2_seq_1    = np.zeros(sequence_len)
            ind_1 = -1
        else:
            done_seq_1[ind_1] = 0

        ind_1 += 1
    
    elapsed = time.time() - start
    sleep_time = UPDATE_INTERVAL - elapsed
    if sleep_time > 0:
        max_duration = sleep_time # seconds, for example

        start_time = time.time()
        while True:
            # If replay buffer has enough data, do a training update
            if replay_buffer.size > batch_size:
                sample = replay_buffer.sample(batch_size)
                agent1.actor.learn(sample)  # same as agent1
            else:
                time.sleep(sleep_time)
                break
            # Check how much time has passed
            elapsed = time.time() - start_time
            if elapsed >= max_duration:
                break
    start = time.time()
    # ---------------
    # AGENT 2 STEP
    # ---------------
    if contact2.inbattle and searching:
        # print(contact2.isBattleMaster)

        contact2.request_tick()
        sct_img_2 = sct.grab(monitor_2)
        frame2 = np.array(sct_img_2)
        frame_bgr_2 = cv2.cvtColor(frame2, cv2.COLOR_BGRA2RGB)

        # Tower health
        tower_viewer_2.update(frame_bgr_2, tower_monitor=health_bars_1)
        # print(frame_bgr_2.shape)
        # Card region for agent2
        card_frame_2 = frame_bgr_2[
            card_monitor_1["top"]:card_monitor_1["top"]+card_monitor_1["height"],
            card_monitor_1["left"]:card_monitor_1["left"]+card_monitor_1["width"]
        ]
        troop_frame_2 = frame_bgr_2[
            yolo_monitor["top"]:yolo_monitor["top"]+yolo_monitor["height"],
            yolo_monitor["left"]:yolo_monitor["left"]+yolo_monitor["width"]
        ]
        resized_troop_2 = cv2.resize(troop_frame_2, (224, 224), interpolation=cv2.INTER_LINEAR)
        resized_troop_2 = np.transpose(resized_troop_2, (2, 0, 1))
        # print(card_frame_2.shape)

        # Elixir detection
        elixir_viewer_2.detect_elixir(card_frame_2)
        elixir_2 = elixir_viewer_2.getElixir()

        # YOLO predictions for troops & deck
        deck_2 = deck_viewer_2.predict(card_frame_2)

        reward_2 = tower_viewer_2.getReward()

        # Observations for agent2
        obs_2 = np.concatenate([
            deck_2,
            tower_viewer_2.getAllHealth(),
            elixir_2,
            [contact2.tick / 720]
        ])
        print(elixir_2)
        print(tower_viewer_1.getAllHealth())
        print(tower_viewer_2.getAllHealth())


        # Store transitions in sequences
        if ind_2 > 0 and ind_2 % sequence_len == 0:
            next_obs_seq_2[ind_2 - 1] = obs_2
            img_seq_2[ind_2] = resized_troop_2

            cur_ad_2 = agent2.actor.calculate_advantage(
                img_seq_2[:-1], img_seq_2[1:],
                obs_seq_2, next_obs_seq_2, reward_seq_2, done_seq_2
            )
            replay_buffer.add_sequence(
                img_seq_2[:-1], img_seq_2[1:],
                obs_seq_2, next_obs_seq_2, action_seq_2,
                reward_seq_2, done_seq_2, cur_ad_2,
                log_prob_seq_2, mask1_seq_2, mask2_seq_2
            )

            # if replay_buffer.size > batch_size:
            #     for _ in range(LEARN_ITER):
            #         sample = replay_buffer.sample(batch_size)
            #         agent2.actor.learn(sample)  # same as agent1

            img_seq_2 = np.zeros((sequence_len + 1, 3, 224, 224))
            obs_seq_2      = np.zeros((sequence_len, obs_dim_2))
            next_obs_seq_2 = np.zeros((sequence_len, obs_dim_2))
            reward_seq_2   = np.zeros(sequence_len)
            action_seq_2   = np.zeros((sequence_len, 2))
            done_seq_2     = np.zeros(sequence_len)
            log_prob_seq_2 = np.zeros(sequence_len)
            mask1_seq_2    = torch.zeros((sequence_len, 9))
            mask2_seq_2    = np.zeros(sequence_len)
            ind_2 = 0

        # Construct masks for agent2
        deckmask1_2 = np.zeros(9, dtype=np.int32)
        deckmask1_2[0] = 1
        deckmask1_2[1:] = deck_2[:]
        deckmask2_2 = [None] * 9
        deckmask2_2[0] = agent2.getBoardMask("troop_noaction")
        for ind in range(len(deck_2)):
            if deck_2[ind] == 1:
                if ind == 1:
                    deckmask2_2[ind+1] = agent2.getBoardMask("spell")
                else:
                    if tower_viewer_2.getAllHealth()[0] == 0:
                        if tower_viewer_2.getAllHealth()[1] == 0:
                            deckmask2_2[ind+1] = agent2.getBoardMask("troop_b")
                        else:
                            deckmask2_2[ind+1] = agent2.getBoardMask("troop_l")
                    elif tower_viewer_2.getAllHealth()[1] == 0:
                        deckmask2_2[ind+1] = agent2.getBoardMask("troop_r")
                    else:
                        deckmask2_2[ind+1] = agent2.getBoardMask("troop_n")
        

        action_2, log_prob_2, m1_2, m2_2 = agent2.getAction(resized_troop_2, obs_2, mask1=deckmask1_2, mask2=deckmask2_2)

        # If there's a non-zero action in the first dimension, place troop, etc.
        if action_2[0] > 0:
            
            troop_num_2 = action_2[0]-1
            x_2 = action_2[1] % 18
            y_2 = action_2[1] // 18
            if contact2.isBattleMaster:
                x_2 = 1000*x_2 + 500
                y_2 = 1000*y_2 + 500
            else:
                x_2 = 17000 - 1000*x_2 + 500
                y_2 = 31000 - 1000*y_2 + 500
            contact2.place_troop(troop_num_2, x_2, y_2)

        # Save data in local buffer
        img_seq_2[ind_2] = resized_troop_2
        obs_seq_2[ind_2]      = obs_2
        reward_seq_2[ind_2]   = reward_2
        action_seq_2[ind_2]   = action_2
        log_prob_seq_2[ind_2] = log_prob_2
        mask1_seq_2[ind_2]    = m1_2
        mask2_seq_2[ind_2]    = m2_2

        if ind_2 > 0:
            next_obs_seq_2[ind_2 - 1] = obs_2

        contact2.check_inbattle()
        if not contact2.inbattle:
            done_seq_2[ind_2] = 1
            next_obs_seq_2[ind_2] = obs_2
            img_seq_2[ind_2+1] = resized_troop_2

            cur_ad_2 = agent2.actor.calculate_advantage(
                img_seq_2[:ind_2+1], img_seq_2[1:ind_2+2],
                obs_seq_2[:ind_2+1], next_obs_seq_2[:ind_2+1],
                reward_seq_2[:ind_2+1], done_seq_2[:ind_2+1]
            )
            replay_buffer.add_sequence(
                img_seq_2[:ind_2+1], img_seq_2[1:ind_2+2],
                obs_seq_2[:ind_2+1], next_obs_seq_2[:ind_2+1],
                action_seq_2[:ind_2+1], reward_seq_2[:ind_2+1], 
                done_seq_2[:ind_2+1], cur_ad_2[:ind_2+1],
                log_prob_seq_2[:ind_2+1],
                mask1_seq_2[:ind_2+1],
                mask2_seq_2[:ind_2+1]
            )

           

            # Reset
            img_seq_2 = np.zeros((sequence_len + 1, 3, 224, 224))
            obs_seq_2      = np.zeros((sequence_len, obs_dim_2))
            next_obs_seq_2 = np.zeros((sequence_len, obs_dim_2))
            reward_seq_2   = np.zeros(sequence_len)
            action_seq_2   = np.zeros((sequence_len, 2))
            done_seq_2     = np.zeros(sequence_len)
            log_prob_seq_2 = np.zeros(sequence_len)
            mask1_seq_2    = torch.zeros((sequence_len, 9))
            mask2_seq_2    = np.zeros(sequence_len)
            ind_2 = -1
            tower_viewer_1.reset()
            tower_viewer_2.reset()
            max_duration = 10.0  # seconds, for example

            start_time = time.time()
            while True:
                # If replay buffer has enough data, do a training update
                if replay_buffer.size > batch_size:
                    sample = replay_buffer.sample(batch_size)
                    agent2.actor.learn(sample)  # same as agent1
                else:
                    time.sleep(10)
                    break
                # Check how much time has passed
                elapsed = time.time() - start_time
                if elapsed >= max_duration:
                    break
            pyautogui.moveTo(350, 1200)
            pyautogui.click()
            max_duration = 5 # seconds, for example

            start_time = time.time()
            while True:
                # If replay buffer has enough data, do a training update
                if replay_buffer.size > batch_size:
                    sample = replay_buffer.sample(batch_size)
                    agent2.actor.learn(sample)  # same as agent1
                else:
                    time.sleep(5)
                    break
                # Check how much time has passed
                elapsed = time.time() - start_time
                if elapsed >= max_duration:
                    break
            pyautogui.moveTo(1100, 1200)
            pyautogui.click()
            pyautogui.moveTo(1800,1299 )
            searching = False
            contact1.inbattle = False
            agent1.actor.save_model()
            max_duration = 10.0  # seconds, for example

            start_time = time.time()
            while True:
                # If replay buffer has enough data, do a training update
                if replay_buffer.size > batch_size:
                    sample = replay_buffer.sample(batch_size)
                    agent2.actor.learn(sample)  # same as agent1
                else:
                    time.sleep(10)
                    break
                # Check how much time has passed
                elapsed = time.time() - start_time
                if elapsed >= max_duration:
                    break

        else:
            done_seq_2[ind_2] = 0

        ind_2 += 1

    elapsed = time.time() - start
    sleep_time = UPDATE_INTERVAL - elapsed
    if sleep_time > 0:
        # time.sleep(sleep_time)
        max_duration = sleep_time  # seconds, for example

        start_time = time.time()
        while True:
            # If replay buffer has enough data, do a training update
            if replay_buffer.size > batch_size:
                sample = replay_buffer.sample(batch_size)
                agent2.actor.learn(sample)  # same as agent1
            else:
                time.sleep(sleep_time)
                break
            # Check how much time has passed
            elapsed = time.time() - start_time
            if elapsed >= max_duration:
                break

