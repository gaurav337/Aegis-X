"""Agent Loop — Dynamic tool orchestration with early stopping.

Generator-based execution with real-time UI feedback via AgentEvent yields.
"""

from typing import Dict, List, Any, Generator
from core.tools.registry import get_registry
from core.early_stopping import EarlyStoppingController, StopReason
from utils.ensemble import EnsembleAggregator
from utils.thresholds import CONFIDENCE_GATE_THRESHOLD, EARLY_STOP_CONFIDENCE, ENSEMBLE_REAL_THRESHOLD, ENSEMBLE_FAKE_THRESHOLD
from core.llm import generate_verdict
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AgentEvent:
    """Real-time progress event for UI streaming."""
    def __init__(self, event_type: str, tool_name: str = None, data: dict = None):
        self.event_type = event_type  # "tool_start", "tool_complete", "verdict", etc.
        self.tool_name = tool_name
        self.data = data or {}


class ForensicAgent:
    """Orchestrates forensic analysis with dynamic tool selection."""
    
    def __init__(self, config):
        self.config = config
        self.registry = get_registry()
        self.ensemble = EnsembleAggregator()
        self.esc = EarlyStoppingController(
            tool_registry=self.registry,
            thresholds=(ENSEMBLE_REAL_THRESHOLD, ENSEMBLE_FAKE_THRESHOLD)
        )
    
    def _run_cpu_phase(self, preprocess_result: Any, media_path: str = None) -> Generator[AgentEvent, Any, bool]:
        """
        Execute CPU tools with confidence gating.
        
        Returns:
            bool: True if confidence gate triggered (skip GPU), False otherwise
        """
        input_data = {
            "media_path": media_path,
            "tracked_faces": preprocess_result.tracked_faces,
            "frames_30fps": preprocess_result.frames_30fps,
            "original_media_type": preprocess_result.original_media_type,
        }
        
        cpu_tools = self.registry.get_cpu_tools()
        
        for tool_name in cpu_tools:
            # ── SKIP LOGIC ──
            if tool_name == "run_rppg" and preprocess_result.original_media_type == "image":
                logger.debug(f"Skipping {tool_name} (static image)")
                continue
            
            if tool_name in ("run_geometry", "run_illumination", "run_corneal"):
                if not preprocess_result.tracked_faces:
                    logger.debug(f"Skipping {tool_name} (no landmarks)")
                    continue
            
            # ── EXECUTE TOOL ──
            yield AgentEvent("tool_start", tool_name)
            
            tool = self.registry.get_tool(tool_name)
            result = tool.execute(input_data)
            
            self.ensemble.add_result(result)
            
            yield AgentEvent("tool_complete", tool_name, data={
                "score": result.score,
                "confidence": result.confidence,
                "evidence_summary": result.evidence_summary,
                "success": result.success,
                "error_msg": result.error_msg
            })
            
            # ── C2PA SHORT-CIRCUIT ──
            if tool_name == "check_c2pa" and result.details.get("c2pa_verified"):
                logger.info("C2PA verified — short-circuiting entire pipeline")
                return True  # Skip everything, go straight to verdict
            
            # ── CONFIDENCE GATE (After CPU phase) ──
            current_score = self.ensemble.get_final_score()
            current_confidence = max(r.confidence for r in self.ensemble.tool_results.values()) if self.ensemble.tool_results else 0.0
            
            if current_confidence >= CONFIDENCE_GATE_THRESHOLD:
                logger.info(f"Confidence gate triggered ({current_confidence:.2f} >= {CONFIDENCE_GATE_THRESHOLD})")
                return True  # Skip GPU phase
        
        return False  # Continue to GPU phase
    
    def _run_gpu_phase(self, preprocess_result: Any) -> Generator[AgentEvent, Any, None]:
        """Execute GPU tools with sequential VRAM cleanup."""
        from utils.vram_manager import run_with_vram_cleanup
        
        input_data = {
            "tracked_faces": preprocess_result.tracked_faces,
            "frames_30fps": preprocess_result.frames_30fps,
        }
        
        gpu_tools = self.registry.get_gpu_tools()
        viable_pending = list(gpu_tools)
        
        for tool_name in gpu_tools:
            if tool_name in viable_pending:
                viable_pending.remove(tool_name)
                
            # ── SKIP LOGIC ──
            if tool_name == "run_sbi":
                visual_result = self.ensemble.tool_results.get("run_univfd")
                from utils.thresholds import SBI_SKIP_UNIVFD_THRESHOLD
                if visual_result and visual_result.score > SBI_SKIP_UNIVFD_THRESHOLD:
                    logger.debug(f"Skipping SBI (UnivFD score > {SBI_SKIP_UNIVFD_THRESHOLD} = fully synthetic)")
                    continue
            
            # ── EXECUTE TOOL ──
            yield AgentEvent("tool_start", tool_name)
            
            tool = self.registry.get_tool(tool_name)
            
            # VRAM-managed execution
            def inference_fn(model):
                return tool.execute(input_data)
            
            result = run_with_vram_cleanup(
                lambda: tool,  # Tool is already loaded via registry
                inference_fn,
                model_name=tool_name,
                required_vram_gb=0.6,
            )
            
            self.ensemble.add_result(result)
            
            yield AgentEvent("tool_complete", tool_name, data={
                "score": result.score,
                "confidence": result.confidence,
                "evidence_summary": result.evidence_summary,
                "success": result.success,
                "error_msg": result.error_msg
            })
            
            # ── EARLY STOP ESTIMATION ──
            # Only include successfully executed tools
            tool_scores = {
                name: res.score for name, res in self.ensemble.tool_results.items()
                if res.success
            }
            
            # We also check for check_c2pa result from earlier
            c2pa_verified = False
            c2pa_res = self.ensemble.tool_results.get("check_c2pa")
            if c2pa_res and c2pa_res.success and c2pa_res.details.get("c2pa_verified"):
                c2pa_verified = True
                
            decision = self.esc.evaluate(
                tool_scores=tool_scores,
                completed_tools=list(self.ensemble.tool_results.keys()),
                c2pa_hardware_verified=c2pa_verified
            )
            
            if decision.should_stop:
                logger.info(f"Early stop triggered: {decision.reason.value} ({decision.confidence:.2f})")
                yield AgentEvent("early_stop", data={"reason": decision.reason.value, "confidence": decision.confidence})
                break
    
    def analyze(self, preprocess_result: Any, media_path: str = None) -> Generator[AgentEvent, Any, dict]:
        """
        Main analysis loop — orchestrates CPU → GPU → Ensemble → LLM.
        
        Yields:
            AgentEvent: Real-time progress updates for UI
        Returns:
            dict: Final verdict with score, confidence, and explanation
        """
        skip_gpu = False
        
        # ── CPU PHASE ──
        for event in self._run_cpu_phase(preprocess_result, media_path):
            yield event
            if event.event_type == "tool_complete" and event.tool_name == "check_c2pa":
                if event.data.get("c2pa_verified"):
                    skip_gpu = True
                    break
        
        # ── GPU PHASE (Conditional) ──
        if not skip_gpu:
            for event in self._run_gpu_phase(preprocess_result):
                yield event
        
        # ── ENSEMBLE SCORING ──
        final_score = self.ensemble.get_final_score()
        verdict = self.ensemble.get_verdict()
        
        # ── LLM SYNTHESIS ──
        yield AgentEvent("llm_start")
        
        explanation = yield from generate_verdict(
            ensemble_score=final_score,
            tool_results=self.ensemble.tool_results,
            verdict=verdict,
        )
        
        yield AgentEvent("verdict", data={
            "verdict": verdict,
            "score": final_score,
            "confidence": max(r.confidence for r in self.ensemble.tool_results.values()),
            "explanation": explanation,
            "tool_count": len(self.ensemble.tool_results),
        })
        
        return {
            "verdict": verdict,
            "score": final_score,
            "explanation": explanation,
        }