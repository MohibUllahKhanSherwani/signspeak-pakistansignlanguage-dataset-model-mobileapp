import subprocess
import sys

try:
    with open("comparison_result.txt", "w") as f:
        # Use sys.executable to ensure we use the same python interpreter (venv)
        subprocess.run([sys.executable, "compare_debug_data.py"], stdout=f, stderr=subprocess.STDOUT)
    print("Comparison finished, check comparison_result.txt")
except Exception as e:
    print(f"Error running comparison: {e}")
