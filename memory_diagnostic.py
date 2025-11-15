#!/usr/bin/env python3
"""
Memory diagnostic script to identify what's consuming memory
"""
import sys
import os

def get_memory():
    """Get current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except:
        return 0

print("=" * 60)
print("MEMORY DIAGNOSTIC")
print("=" * 60)

# Baseline
baseline = get_memory()
print(f"\n1. Baseline (Python only): {baseline:.2f} MB")

# After importing common libraries
import json
import tempfile
step2 = get_memory()
print(f"2. After stdlib imports: {step2:.2f} MB (delta: +{step2-baseline:.2f} MB)")

# Check if sentence-transformers is still installed
try:
    import sentence_transformers
    st_mem = get_memory()
    print(f"3. WARNING: sentence-transformers imported: {st_mem:.2f} MB (delta: +{st_mem-step2:.2f} MB)")
    print("   -> This should NOT be installed! Run: pip uninstall sentence-transformers -y")
except ImportError:
    print(f"3. sentence-transformers: NOT installed (good!)")

# Check if torch is still installed
try:
    import torch
    torch_mem = get_memory()
    print(f"4. WARNING: PyTorch imported: {torch_mem:.2f} MB (delta: +{torch_mem-step2:.2f} MB)")
    print("   -> This is heavy! Run: pip uninstall torch torchvision torchaudio -y")
except ImportError:
    print(f"4. PyTorch: NOT installed (good!)")

# Check psutil
try:
    import psutil
    ps_mem = get_memory()
    print(f"5. psutil imported: {ps_mem:.2f} MB (delta: +{ps_mem-step2:.2f} MB)")
except ImportError:
    print(f"5. psutil: NOT installed")

# Check faster-whisper
try:
    from faster_whisper import WhisperModel
    fw_mem = get_memory()
    print(f"6. faster-whisper imported: {fw_mem:.2f} MB (delta: +{fw_mem-step2:.2f} MB)")
except ImportError:
    print(f"6. faster-whisper: NOT installed")
    print("   -> Run: pip install faster-whisper")

# Check openai-whisper
try:
    import whisper
    ow_mem = get_memory()
    print(f"7. WARNING: openai-whisper imported: {ow_mem:.2f} MB (delta: +{ow_mem-step2:.2f} MB)")
    print("   -> This is heavy! Consider using faster-whisper instead")
except ImportError:
    print(f"7. openai-whisper: NOT installed (good if using faster-whisper)")

# Load the actual model
try:
    print(f"\n8. Loading Whisper model...")
    from faster_whisper import WhisperModel
    before_model = get_memory()
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    after_model = get_memory()
    print(f"   Model loaded: {after_model:.2f} MB (delta: +{after_model-before_model:.2f} MB)")
    print(f"   Note: 'tiny' model is smallest. 'base' is larger.")
except Exception as e:
    print(f"   Could not load model: {e}")

print("\n" + "=" * 60)
print("INSTALLED PACKAGES:")
print("=" * 60)
import subprocess
result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
relevant_packages = [line for line in result.stdout.split('\n') if any(x in line.lower() for x in ['torch', 'whisper', 'transform', 'onnx', 'sentencepiece'])]
if relevant_packages:
    for pkg in relevant_packages:
        print(pkg)
else:
    print("No AI/ML packages found")

print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

current = get_memory()
if current > 500:
    print("Memory usage is HIGH (>500 MB). Possible causes:")
    print("1. PyTorch is still installed (uninstall it)")
    print("2. sentence-transformers is still installed (uninstall it)")
    print("3. Using 'base' or larger Whisper model (use 'tiny' for testing)")
    print("4. openai-whisper instead of faster-whisper (switch to faster-whisper)")
elif current > 300:
    print("Memory usage is MODERATE (300-500 MB):")
    print("1. This might be 'base' Whisper model - consider 'tiny' for lower memory")
    print("2. Check if any PyTorch dependencies remain")
else:
    print("Memory usage is GOOD (<300 MB)!")
    print("This is expected for faster-whisper with tiny/base models")

print("\nTo reduce memory further:")
print("- Use 'tiny' model instead of 'base': WhisperModel('tiny', ...)")
print("- Only load model when needed (lazy loading)")
print("- Make sure torch/sentence-transformers are completely removed")