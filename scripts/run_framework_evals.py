"""Run all 5 framework evals sequentially."""
import subprocess, time, sys, os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

frameworks = ["litellm", "openai", "langchain", "crewai", "dspy"]

for fw in frameworks:
    print(f"\n{'='*60}")
    print(f"Running: {fw}")
    print(f"{'='*60}")
    config = f"examples/phoenix_auto_trace/eval_{fw}.yaml"
    start = time.monotonic()
    result = subprocess.run(
        ["uv", "run", "p2m", "run", "--config", config],
        capture_output=True, text=True
    )
    elapsed = time.monotonic() - start
    # Print stderr (stage progress)
    for line in result.stderr.split("\n"):
        if any(kw in line for kw in ["done", "fail", "pipeline", "Error"]):
            print(f"  {line.strip()}")
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"  {fw}: {elapsed:.0f}s ({status})")
