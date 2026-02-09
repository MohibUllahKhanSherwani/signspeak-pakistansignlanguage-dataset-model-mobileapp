from pydantic import BaseModel
from typing import List

class SequenceData(BaseModel):
    # A list of 30 frames, where each frame is a list of 126 floats (landmarks)
    landmarks: List[List[float]]