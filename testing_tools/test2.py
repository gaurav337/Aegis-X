import numpy as np
from core.tools.corneal_tool import CornealTool

import numpy as np

tool = CornealTool()
tool.setup()


face1 = {
    "face_crop_224": np.zeros((224, 224, 3), dtype=np.uint8),
    "landmarks": np.zeros((478, 2), dtype=np.float32),
    "trajectory_bboxes": {0: (0, 0, 224, 224)},
    "best_frame_idx": 0,
    "identity_id": 1
}

# Add realistic catchlights
face1["face_crop_224"][100, 90] = [255, 255, 255]
face1["face_crop_224"][100, 130] = [255, 255, 255]

# Iris landmarks (full frame coords)
face1["landmarks"][468] = [90, 100]
face1["landmarks"][473] = [130, 100]

face2 = {
    "face_crop_224": np.zeros((224, 224, 3), dtype=np.uint8),
    "landmarks": np.zeros((478, 2), dtype=np.float32),
    "trajectory_bboxes": {0: (0, 0, 224, 224)},
    "best_frame_idx": 0,
    "identity_id": 2
}

# Asymmetric catchlights
face2["face_crop_224"][100, 90] = [255, 255, 255]
face2["face_crop_224"][105, 130] = [255, 255, 255]

face2["landmarks"][468] = [90, 100]
face2["landmarks"][473] = [130, 100]

face3 = {
    "face_crop_224": np.zeros((224, 224, 3), dtype=np.uint8),
    "landmarks": np.zeros((478, 2), dtype=np.float32),
    "trajectory_bboxes": {0: (0, 0, 224, 224)},
    "best_frame_idx": 0,
    "identity_id": 3
}

# No bright pixels → no catchlights

face3["landmarks"][468] = [90, 100]
face3["landmarks"][473] = [130, 100]

tracked_faces = [face1, face2, face3]

result = tool._run_inference({
    "tracked_faces": tracked_faces
})

print(result)
