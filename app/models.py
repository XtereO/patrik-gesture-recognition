from pydantic import BaseModel

class RecognitionReturn(BaseModel):
    right_hand: list[list[float, float, float]]
    left_hand: list[list[float, float, float]]
    pose: list[list[float, float, float]]
