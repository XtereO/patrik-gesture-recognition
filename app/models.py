from pydantic import BaseModel

class RecognitionBody(BaseModel):
    data: list[list[list[int, int, int]]]
    hi: int

class RecognitionReturn(BaseModel):
    right_hand: list[list[float, float, float]]
    left_hand: list[list[float, float, float]]
    pose: list[list[float, float, float]]
