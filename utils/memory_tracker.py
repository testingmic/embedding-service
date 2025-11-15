"""
Memory tracking utilities for monitoring resource usage
"""
import os
from typing import Dict, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[WARNING] psutil not installed. Memory tracking will be limited.")
    print("   Install with: pip install psutil")


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory statistics in MB:
        - process_memory: Current process memory usage (MB)
        - system_memory_used: System memory used (MB)
        - system_memory_total: Total system memory (MB)
        - system_memory_percent: System memory usage percentage
    """
    if not PSUTIL_AVAILABLE:
        # Fallback: return minimal info without psutil
        return {
            "process_memory_mb": 0.0,
            "system_memory_used_mb": 0.0,
            "system_memory_total_mb": 0.0,
            "system_memory_percent": 0.0
        }
    
    process = psutil.Process(os.getpid())
    
    # Process memory
    process_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    
    # System memory
    system_memory = psutil.virtual_memory()
    
    return {
        "process_memory_mb": round(process_memory, 2),
        "system_memory_used_mb": round(system_memory.used / (1024 * 1024), 2),
        "system_memory_total_mb": round(system_memory.total / (1024 * 1024), 2),
        "system_memory_percent": round(system_memory.percent, 2)
    }


def get_memory_delta(start_memory: Optional[float], end_memory: Optional[float]) -> Optional[float]:
    """
    Calculate memory delta between two measurements.
    
    Args:
        start_memory: Starting memory in MB
        end_memory: Ending memory in MB
        
    Returns:
        Memory delta in MB, or None if either value is None
    """
    if start_memory is None or end_memory is None:
        return None
    return round(end_memory - start_memory, 2)


def log_memory_usage(operation: str, memory_stats: Dict[str, float]) -> None:
    """
    Log memory usage for an operation.
    
    Args:
        operation: Name of the operation
        memory_stats: Memory statistics dictionary
    """
    print(f"[MEMORY] Memory usage for {operation}:")
    print(f"   Process: {memory_stats['process_memory_mb']} MB")
    print(f"   System: {memory_stats['system_memory_used_mb']} MB / {memory_stats['system_memory_total_mb']} MB ({memory_stats['system_memory_percent']}%)")