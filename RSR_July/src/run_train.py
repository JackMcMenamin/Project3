"""
Crash Wrapper Script for RSR Training

A simple utility script used during development to catch unhandled exceptions 
(e.g., Syntax errors, CUDA OOMs, import failures) and dump the exact stack 
trace into a local text file (`crash.txt`) for immediate debugging.

This is especially useful when executing via background automated tasks 
where standard error output might be buffered or truncated.
"""

import traceback
import sys

if __name__ == "__main__":
    try:
        # Attempt to import and execute the main training logic
        import train
        train.main()
    except Exception as e:
        # On any failure, catch it and dump the stack trace cleanly to disk
        with open("crash.txt", "w") as f:
            traceback.print_exc(file=f)
        print("CRASHED:", e)
