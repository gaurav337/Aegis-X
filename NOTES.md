## Day 12: SBI Tool - Weights Status

✅ Tool architecture complete and tested
⚠️ Using random initialization (ImageNet backbone only)

### For Production:
- Need official SBI weights from mapooon/SelfBlendedImages
- Archive format is torch sharded (data.pkl + data/ directory)
- Requires custom extraction script

### Alternatives:
1. Find pre-trained weights on HuggingFace
2. Train on Google Colab (free tier)
3. Use FreqNet as primary detector (CPU-friendly)

### Current Impact:
- SBI scores are placeholder values (~0.5 random)
- Architecture fully functional for development
- Ensemble logic can be tested with mock scores
