from core.tools.freqnet_tool import FreqNetTool
import cv2

tool = FreqNetTool()
tool.setup()

img = cv2.imread("test.jpg")

input_data = {
    "tracked_faces": [
        {
            "face_crop_224": cv2.resize(img, (224, 224)),
            "landmarks": [[50,50]] * 478  # dummy landmarks (needed)
        }
    ]
}

result = tool._run_inference(input_data)
print(result)