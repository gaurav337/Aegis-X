from core.tools.dct_tool import DCTTool
import cv2

tool = DCTTool()
tool.setup()


img = cv2.imread("test.jpg")

# First compression
cv2.imwrite("compressed1.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])

# Second compression
img2 = cv2.imread("compressed1.jpg")
cv2.imwrite("compressed2.jpg", img2, [cv2.IMWRITE_JPEG_QUALITY, 50])

for file in ["test.jpg", "compressed2.jpg"]:
    img = cv2.imread(file)

    result = tool._run_inference({
        "tracked_faces": [{"face_crop_224": cv2.resize(img, (224,224))}]
    })

    print(file, result.details, result.score)