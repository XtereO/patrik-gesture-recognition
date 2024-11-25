import os
import sys
import base64
import logging

import numpy as np

import cv2
import mediapipe as mp
from fastapi import FastAPI

from .utils import add_padding, reduce_landmarks
from .models import RecognitionReturn, RecognitionBody

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)
os.chdir(dname)

logger = logging.getLogger('uvicorn.error')

mp_holistic = mp.solutions.holistic

app = FastAPI()

@app.post("/recognition")
async def recognition(body: RecognitionBody):
    img_base64 = body.base64
    decoded = add_padding(base64.b64decode(img_base64))
    buffer = np.fromstring(decoded, np.float32)
    image_array = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        data = np.array(image_array).astype(np.uint8)

        res = holistic.process(data)

        return RecognitionReturn(right_hand = reduce_landmarks(res.right_hand_landmarks), 
                                 left_hand = reduce_landmarks(res.left_hand_landmarks),
                                 pose = reduce_landmarks(res.pose_landmarks))