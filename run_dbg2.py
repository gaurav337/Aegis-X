import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tools.rppg_tool import RPPGTool
import numpy as np

tool = RPPGTool()
fake_frames = []
fake_frames.append(np.ones((480, 640, 3), dtype=np.uint8) * 128)
for _ in range(89):
    fake_frames.append(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
dummy_trajectory = {i: (100, 100, 300, 300) for i in range(90)}
dummy_tracked_faces = [{"identity_id": 0, "landmarks": np.zeros((478, 2)), "trajectory_bboxes": dummy_trajectory}] 

# Let's mock the tool method to print debug values
original_run = tool._run_inference

def debug_run(input_data):
    res = original_run(input_data)
    return res

result = debug_run({"frames_30fps": fake_frames, "tracked_faces": dummy_tracked_faces})
print(result)
