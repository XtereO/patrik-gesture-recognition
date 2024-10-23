import os
import sys
import logging

import numpy as np
import mediapipe as mp

from fastapi import FastAPI

from .models import RecognitionBody, RecognitionReturn

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)
os.chdir(dname)

logger = logging.getLogger('uvicorn.error')

mp_holistic = mp.solutions.holistic

app = FastAPI()

def reduce_landmarks(items):
    if(items is None):
        return []
    
    acc = []
    for i in items.landmark:
        acc.append([i.x, i.y, i.z])
    return acc

@app.post("/recognition/")
async def recognition(body: RecognitionBody) -> RecognitionReturn:
   
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        dat = np.array(body.data).astype(np.uint8)
        res = holistic.process(dat)

        return RecognitionReturn(right_hand = reduce_landmarks(res.right_hand_landmarks), 
                                 left_hand = reduce_landmarks(res.left_hand_landmarks),
                                 pose = reduce_landmarks(res.pose_landmarks))