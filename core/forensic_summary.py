"""Forensic Summary — Tool outputs → Structured Phi-3 prompt.

Converts ensemble results into a natural language prompt that grounds
every claim in specific tool evidence. The LLM never sees raw pixels.
"""

from typing import Dict
from core.data_types import ToolResult
from utils.thresholds import REAL_THRESHOLD, FAKE_THRESHOLD


def build_phi3_prompt(
    ensemble_score: float,
    tool_results: Dict[str, ToolResult],
    verdict: str,
) -> str:
    """
    Build structured prompt for Phi-3 Mini reasoning.
    
    Per Spec Section 5: LLM receives only structured text, never raw images.
    """
    prompt_parts = []
    
    # ── HEADER ──
    prompt_parts.append("=== AEGIS-X FORENSIC ANALYSIS ===\n")
    prompt_parts.append(f"Ensemble Score: {ensemble_score:.3f}")
    prompt_parts.append(f"Verdict: {verdict}\n")
    
    # ── TOOL EVIDENCE (Partitioned by Identity) ──
    prompt_parts.append("=== TOOL EVIDENCE ===\n")
    
    # C2PA
    if "check_c2pa" in tool_results:
        r = tool_results["check_c2pa"]
        if r.details.get("c2pa_verified"):
            prompt_parts.append(f"✅ C2PA: Cryptographically verified (signer: {r.details.get('signer', 'Unknown')})")
        else:
            prompt_parts.append("⚪ C2PA: No provenance data found (unsigned media)")
    
    # rPPG
    if "run_rppg" in tool_results:
        r = tool_results["run_rppg"]
        label = r.details.get("liveness_label", "UNKNOWN")
        prompt_parts.append(f"💓 rPPG: {label} — {r.evidence_summary}")
    
    # DCT
    if "run_dct" in tool_results:
        r = tool_results["run_dct"]
        prompt_parts.append(f"📐 DCT: peak_ratio={r.details.get('peak_ratio', 0):.3f}, score={r.score:.3f}")
        if r.details.get("grid_artifacts"):
            prompt_parts.append("  → Double-quantization artifacts detected (compression or tampering)")
    
    # GEOMETRY ← ADD THIS
    if "run_geometry" in tool_results:
        r = tool_results["run_geometry"]
        violations = r.details.get("violations", [])
        prompt_parts.append(f"📏 Geometry: score={r.score:.3f} ({len(violations)}/7 checks failed)")
        if violations:
            prompt_parts.append(f"  → Violations: {', '.join(violations)}")
        else:
            prompt_parts.append("  → All anatomical ratios within normal human range")
    
    # Illumination
    if "run_illumination" in tool_results:
        r = tool_results["run_illumination"]
        prompt_parts.append(f"💡 Illumination: {r.evidence_summary}")
    
    # Corneal
    if "run_corneal" in tool_results:
        r = tool_results["run_corneal"]
        prompt_parts.append(f"👁️ Corneal: {r.evidence_summary}")
    
    # UnivFD
    if "run_univfd" in tool_results:
        r = tool_results["run_univfd"]
        prompt_parts.append(f"🧠 UnivFD: score={r.score:.3f}")

    # Xception
    if "run_xception" in tool_results:
        r = tool_results["run_xception"]
        prompt_parts.append(f"🔬 XceptionNet: score={r.score:.3f}")
    
    # SBI
    if "run_sbi" in tool_results:
        r = tool_results["run_sbi"]
        if r.details.get("boundary_detected"):
            prompt_parts.append(f"🪡 SBI: Blend boundary at {r.details.get('boundary_region', 'unknown')} (score={r.score:.3f})")
        else:
            prompt_parts.append(f"🪡 SBI: No blend boundary detected (score={r.score:.3f})")
    
    # FreqNet
    if "run_freqnet" in tool_results:
        r = tool_results["run_freqnet"]
        prompt_parts.append(f"📡 FreqNet: score={r.score:.3f}")
        if r.details.get("anomaly_region"):
            prompt_parts.append(f"  → Anomaly: {r.details['anomaly_region']}")
    
    # ── REASONING RULES ──
    prompt_parts.append("\n=== REASONING RULES ===")
    prompt_parts.append("1. Ground every claim in specific tool output (e.g., 'DCT peak_ratio=0.86 indicates...')")
    prompt_parts.append("2. CRITICAL SCORING LOGIC: A score closer to 0.0 means AUTHENTIC/REAL. A score closer to 1.0 means FAKE/TAMPERED. Do not get this backwards! A low score is a passing grade for authenticity.")
    prompt_parts.append("3. Explain conflicts between tools (e.g., 'rPPG detected pulse but UnivFD flagged semantics...')")
    prompt_parts.append("4. Use probabilistic language ('suggests', 'consistent with') — avoid absolute certainty")
    prompt_parts.append("5. Keep explanation under 150 words like in lamen language")
    
    # ── OUTPUT FORMAT ──
    prompt_parts.append("\n=== OUTPUT FORMAT ===")
    prompt_parts.append("Return a single paragraph of plain text containing your explanation. DO NOT use JSON, markdown, or any structured format. Just write the natural language explanation directly.")
    
    return "\n".join(prompt_parts)


async def generate_verdict(
    ensemble_score: float,
    tool_results: Dict[str, ToolResult],
    verdict: str,
) -> str:
    """
    Generate LLM verdict with streaming support.
    
    Returns:
        str: Natural language explanation from Phi-3
    """
    from core.llm import stream_completion
    
    prompt = build_phi3_prompt(ensemble_score, tool_results, verdict)
    
    explanation = ""
    async for token in stream_completion(prompt, temperature=0.1, max_tokens=512):
        explanation += token
        # Yield token for UI streaming (handled by agent.py)
    
    return explanation