#!/usr/bin/env python3
"""Day 14 Pre-Implementation Diagnostics"""

import importlib
import inspect

print("=" * 60)
print("DAY 14 PRE-IMPLEMENTATION DIAGNOSTICS")
print("=" * 60)

# ── Diagnostic 1: Tool Imports ──
print("\n1️⃣  VERIFYING ALL 9 TOOLS...\n")

TOOL_SPECS = [
    ("core.tools.c2pa_tool",        "C2PATool"),
    ("core.tools.rppg_tool",        "RPPGTool"),
    ("core.tools.dct_tool",         "DCTTool"),
    ("core.tools.geometry_tool",    "GeometryTool"),
    ("core.tools.illumination_tool","IlluminationTool"),
    ("core.tools.corneal_tool",     "CornealTool"),
    ("core.tools.clip_adapter_tool","ClipAdapterTool"),
    ("core.tools.sbi_tool",        "SBITool"),
    ("core.tools.freqnet_tool",    "FreqNetTool"),
]

tool_names = []
issues = []

for module_path, class_name in TOOL_SPECS:
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        instance = cls()
        
        tool_name = getattr(instance, 'tool_name', 'MISSING')
        tool_names.append(tool_name)
        
        has_execute = hasattr(instance, 'execute') and callable(instance.execute)
        has_setup = hasattr(instance, 'setup') and callable(instance.setup)
        has_gpu_attr = hasattr(instance, 'requires_gpu')
        
        status = "✅" if (has_execute and tool_name != 'MISSING') else "❌"
        print(f"{status} {class_name:25} → {tool_name}")
        
        if not has_setup:
            print(f"   ⚠️  Missing setup() method")
        if not has_gpu_attr and tool_name in ['run_clip_adapter', 'run_sbi', 'run_freqnet']:
            print(f"   ⚠️  Missing requires_gpu attribute")
            
    except Exception as e:
        print(f"❌ {class_name:25} → {type(e).__name__}: {e}")
        issues.append((class_name, str(e)))

# ── Diagnostic 2: ToolResult Schema ──
print("\n2️⃣  VERIFYING ToolResult SCHEMA...\n")

try:
    from core.data_types import ToolResult
    sig = inspect.signature(ToolResult.__init__)
    params = list(sig.parameters.keys())
    
    required = ['tool_name', 'score', 'confidence', 'success', 'error_msg', 'execution_time']
    missing = [f for f in required if f not in params]
    
    print(f"   Fields: {', '.join(params)}")
    
    if missing:
        print(f"   ❌ MISSING: {missing}")
        issues.append(("ToolResult", f"Missing fields: {missing}"))
    else:
        print(f"   ✅ All required fields present")
        
except Exception as e:
    print(f"   ❌ Failed to import ToolResult: {e}")
    issues.append(("ToolResult", str(e)))

# ── Summary ──
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\nTools verified: {len(tool_names)}/9")
print(f"Issues found: {len(issues)}")

if issues:
    print("\n⚠️  FIX THESE BEFORE BUILDING REGISTRY:\n")
    for name, issue in issues:
        print(f"   • {name}: {issue}")
    print("\n❌ DO NOT PROCEED - fix issues first!")
else:
    print("\n✅ ALL DIAGNOSTICS PASSED")
    print("\n🚀 READY TO BUILD REGISTRY")

print()
