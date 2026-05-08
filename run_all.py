"""
Run all Day 22 lab steps sequentially, or a single specific step.
Usage:
  python run_all.py          # Runs all steps (1, 2, 3, 4)
  python run_all.py --step 3 # Runs only step 3
"""

import sys
import subprocess
import argparse


def run_script(script_name):
    """Run a Python script and print output in real-time."""
    print("\n" + "=" * 60)
    print(f"🚀 Running {script_name}...")
    print("=" * 60 + "\n")

    try:
        # Use the virtual environment's python interpreter to run the script
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            text=True
        )
        print(f"\n✅ {script_name} completed successfully!\n")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}: {e}\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run Day 22 Lab scripts.")
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run only a specific step (1, 2, 3, or 4)"
    )
    args = parser.parse_args()

    # Mapping of steps to python script filenames
    steps = {
        1: "01_langsmith_rag_pipeline.py",
        2: "02_prompt_hub_ab_routing.py",
        3: "03_ragas_evaluation.py",
        4: "04_guardrails_validator.py"
    }

    if args.step:
        script = steps[args.step]
        run_script(script)
    else:
        print("🌟 Starting full Day 22 Lab execution...")
        for step, script in steps.items():
            run_script(script)
        print("🎉 All Day 22 Lab steps completed!")


if __name__ == "__main__":
    main()
