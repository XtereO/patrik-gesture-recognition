from pydantic import BaseModel

class RecognitionBody(BaseModel):
    base64: str

class RecognitionReturn(BaseModel):
    right_hand: list[list[float, float, float]]
    left_hand: list[list[float, float, float]]
    pose: list[list[float, float, float]]
