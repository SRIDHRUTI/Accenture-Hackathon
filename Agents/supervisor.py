#!/usr/bin/env python3
import os
import subprocess
import argparse
import pandas as pd
import sys

def run_agent(script, args_list=[]):
    """
    Executes the given agent script with its default arguments.
    Forces the Transformers library to use PyTorch.
    """
    env = os.environ.copy()
    env["TRANSFORMERS_NO_TF"] = "1"

    command = [sys.executable, script] + args_list
    print("Coordinator: Running:", " ".join(command))
    subprocess.run(command, check=True, env=env)
    print("Coordinator: Completed", script, "\n")

def generate_final_csv():
    """
    Loads outputs from existing agents, aggregates top candidates, and saves final CSV
    expected by the dashboard (final_selected_candidates.csv).
    """
    try:
        scores_df = pd.read_csv("cv_grading_results.csv")
        persona_df = pd.read_csv("persona_fit_results.csv")
        bias_df = pd.read_csv("cv_bias_fairness.csv")
        explain_df = pd.read_csv("explainability_results.csv")

        merged = scores_df.copy()
        merged["persona_fit_score"] = persona_df["Persona_Score"]
        merged["cv_bias_flags"] = bias_df["Bias_Flags"]
        merged["explanation"] = explain_df["Explanation"]

        # Normalizing to compute final score (can be improved further)
        merged["updated_score"] = (
            0.4 * merged["CV_Score"] +
            0.3 * merged["persona_fit_score"] +
            0.3 * (1 - merged["cv_bias_flags"].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0) / 10)
        )

        merged.rename(columns={
            "Candidate": "candidate_filename",
            "CV_Score": "grade_score"
        }, inplace=True)

        merged.to_csv("final_selected_candidates.csv", index=False)
        print("✅ Final candidate selection file generated: final_selected_candidates.csv")

    except Exception as e:
        print(f"❌ Error while generating final output CSV: {e}")

def main(args):
    agents = [
        "jd_optimizer.py",
        "cv_grader.py",
        "bias_agent.py",
        "persona_agent.py",
        "explainability_agent.py",
        "feedback_agent.py",      # Optional, include only if implemented
        "sql_agent.py"            # Ensure this updates the SQLite DB correctly
    ]

    for script in agents:
        run_agent(script)

    print("✅ Running final aggregation step...")
    generate_final_csv()

    final_df = pd.read_csv(args.final_selected, encoding="utf-8")
    print("\nCoordinator: Final Selected Candidates:")
    print(final_df[["candidate_filename", "updated_score", "grade_score", "persona_fit_score"]].head(10).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coordinator Agent for Full Recruitment Pipeline")
    parser.add_argument("--final_selected", type=str, default="final_selected_candidates.csv",
                        help="Final output CSV from SQLite Memory Agent (default: final_selected_candidates.csv)")
    args = parser.parse_args()
    main(args)
