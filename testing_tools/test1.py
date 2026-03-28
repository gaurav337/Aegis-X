from core.tools.c2pa_tool import C2PATool
from pathlib import Path

tool = C2PATool()
tool.setup()

result = tool._run_inference({
    "media_path": "test1.jpg"   # put your file here
})

print(result)