import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config import AegisConfig
from utils.preprocessing import Preprocessor
from core.tools.registry import get_registry

app = FastAPI(title="Aegis-X Web Interface")

# Initialize config and registry once
config = AegisConfig()
preprocessor = Preprocessor(config)
registry = get_registry()
cpu_tools = ["check_c2pa", "run_dct", "run_geometry", "run_illumination"]

UPLOAD_DIR = Path("downloads/web_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/api/analyze")
async def analyze_media(file: UploadFile = File(...)):
    try:
        # Save temp file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Preprocessing
        prep_result = preprocessor.process_media(file_path)
        if not prep_result.has_face:
            return JSONResponse(status_code=400, content={"error": "No faces detected in the media.", "success": False})
            
        input_data = {
            "tracked_faces": prep_result.tracked_faces,
            "frames_30fps": prep_result.frames_30fps,
            "original_media_type": prep_result.original_media_type,
        }

        # Run Tools
        results = []
        for tool_name in cpu_tools:
            res = registry.execute_tool(tool_name, input_data)
            results.append({
                "tool_name": tool_name,
                "success": res.success,
                "score": float(res.score) if res.score is not None else 0.0,
                "confidence": float(res.confidence) if res.confidence is not None else 0.0,
                "evidence_summary": res.evidence_summary,
                "error_msg": res.error_msg
            })
            
        # Optional: compute final avg score
        scores = [r["score"] for r in results if r["success"] and r["tool_name"] != "check_c2pa"]
        final_score = sum(scores) / len(scores) if scores else 0.0
        is_fake = final_score > 0.50
            
        return {
            "success": True, 
            "results": results, 
            "final_score": final_score, 
            "is_fake": is_fake,
            "faces_detected": len(prep_result.tracked_faces)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "success": False})

# Mount frontend
app.mount("/", StaticFiles(directory="web", html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    # run with `python run_web.py`
    uvicorn.run("run_web:app", host="0.0.0.0", port=8000, reload=True)
