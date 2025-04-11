#!/usr/bin/env python3
import os
import subprocess
import argparse
import pandas as pd
import sys  # ✅ Required to use correct Python interpreter

def run_agent(script, args_list=[]):
    """
    Executes the given agent script with its default arguments.
    Forces the Transformers library to use PyTorch.
    """
    env = os.environ.copy()
    env["TRANSFORMERS_NO_TF"] = "1"

    # ✅ Use the current Python interpreter to avoid missing modules like 'pandas'
    command = [sys.executable, script] + args_list

    print("Coordinator: Running:", " ".join(command))
    subprocess.run(command, check=True, env=env)
    print("Coordinator: Completed", script, "\n")

def main(args):
    # List of agents to run sequentially
    agents = [
        "jd_optimizer.py",               # JD Extractor + Optimizer Agent
        "cv_grader.py",                  # CV Parser + Grader Agent
        "bias_agent.py",                 # Bias & Fairness Monitor Agent
        "persona_agent.py",              # Persona-Fit Agent
        "explainability_agent.py",       # Explainability Agent
        "feedback_agent.py",             # Recruiter Feedback Agent
        "sql_agent.py"                   # SQLite Memory Agent
    ]

    # Run each agent script
    for script in agents:
        run_agent(script)

    # Load and print the final selected candidates
    final_df = pd.read_csv(args.final_selected, encoding="utf-8")
    print("\nCoordinator: Final Selected Candidates:")
    print(final_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coordinator Agent for Full Recruitment Pipeline")
    parser.add_argument("--final_selected", type=str, default="final_selected_candidates.csv",
                        help="Final output CSV from SQLite Memory Agent (default: final_selected_candidates.csv)")
    args = parser.parse_args()
    main(args)
