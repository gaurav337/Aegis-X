import json
from pathlib import Path
from typing import Any, Dict

from core.base_tool import BaseForensicTool
from core.data_types import ToolResult


class C2PATool(BaseForensicTool):
    """Tool for verifying C2PA Content Credentials provenance data."""

    @property
    def tool_name(self) -> str:
        return "check_c2pa"

    def setup(self) -> None:
        """Import verification for c2pa-python."""
        # FIX: Catch the import error here so it doesn't crash the pipeline at boot
        try:
            import c2pa
            self._c2pa_available = True
        except ImportError:
            self._c2pa_available = False
            # You could also inject a logger here to warn that the tool is degraded

    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        """Run C2PA extraction and verification logic."""
        if "media_path" not in input_data:
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                score=0.5,
                confidence=0.0,
                details={},
                error=True,
                error_msg="Missing media_path in input_data",
                execution_time=0.0,
                evidence_summary="Missing media_path"
            )

        # FIX: Rely on the setup flag rather than re-importing blindly
        if getattr(self, "_c2pa_available", False) is False:
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                score=0.5,
                confidence=0.0,
                details={},
                error=True,
                error_msg="c2pa-python library not available",
                execution_time=0.0,
                evidence_summary="Missing library"
            )

        media_path = str(input_data["media_path"])
        import c2pa
        
        try:
            c2pa_dict = None
            
            # Attempt to use the newer c2pa.Reader API, fallback to c2pa.read_file if present
            if hasattr(c2pa, "read_file"):
                # Older API version
                try:
                    c2pa_data = c2pa.read_file(media_path)
                    if not c2pa_data:
                        return self._no_c2pa_result()
                        
                    if isinstance(c2pa_data, str):
                        c2pa_dict = json.loads(c2pa_data)
                    else:
                        c2pa_dict = c2pa_data
                except Exception as read_err:
                    # FIX: Apply the same "no jumbf" grace to the old API
                    err_msg = str(read_err).lower()
                    if "not found" in err_msg or "no jumbf" in err_msg or "not supported" in err_msg:
                         return self._no_c2pa_result()
                    raise read_err
            else:
                # Newer Reader API (v0.2+)
                try:
                    reader = c2pa.Reader(media_path)
                    json_str = reader.json()
                    
                    # FIX: Prevent JSONDecodeError on empty returns
                    if not json_str:
                        return self._no_c2pa_result()
                        
                    c2pa_dict = json.loads(json_str) if isinstance(json_str, str) else json_str
                except Exception as read_err:
                    err_msg = str(read_err).lower()
                    if "not found" in err_msg or "no jumbf" in err_msg or "not supported" in err_msg or "notsupported" in err_msg:
                         return self._no_c2pa_result()
                    raise read_err
            
            # Extract signer and timestamp safely
            signer = "Unknown Signer"
            timestamp = "Unknown Timestamp"
            
            if c2pa_dict and "active_manifest" in c2pa_dict:
                manifest_claim = c2pa_dict.get("manifests", {}).get(c2pa_dict["active_manifest"], {})
                
                sig_info = manifest_claim.get("signature_info", {})
                if "issuer" in sig_info:
                    signer = sig_info["issuer"]
                if "time" in sig_info:
                    timestamp = sig_info["time"]
            else:
                # Edge case: parsed valid JSON but it lacks actual manifest data
                return self._no_c2pa_result()
            
            # FIX: Lower confidence if we couldn't cryptographically identify the signer
            confidence_score = 1.0 if signer != "Unknown Signer" else 0.4

            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                score=0.0,
                confidence=confidence_score,
                details=c2pa_dict,
                error=False,
                error_msg=None,
                execution_time=0.0,
                evidence_summary=f"Signed by {signer} at {timestamp}"
            )

        except json.JSONDecodeError as je:
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                score=0.5,
                confidence=0.0,
                details={},
                error=True,
                error_msg=f"Failed to parse C2PA JSON: {str(je)}",
                execution_time=0.0,
                evidence_summary="C2PA metadata is malformed."
            )
        except Exception as e:
            # Fallback for unexpected errors
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                score=0.5,
                confidence=0.0,
                details={},
                error=True,
                error_msg=str(e),
                execution_time=0.0,
                evidence_summary=f"C2PA read error: {str(e)}"
            )

    def _no_c2pa_result(self) -> ToolResult:
        """Helper to return an abstention result when no C2PA data is present."""
        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=0.0,
            confidence=0.0,
            details={},
            error=False,
            error_msg=None,
            execution_time=0.0,
            evidence_summary="No C2PA provenance data found."
        )