#!/usr/bin/env python3
import os
import subprocess
import argparse
import pandas as pd
import sys

def run_agent(script, args_list=[]):
    """
    Executes the given agent script and logs output/errors clearly.
    """
    env = os.environ.copy()
    env["TRANSFORMERS_NO_TF"] = "1"
    command = [sys.executable, script] + args_list

    print(f"\n🔄 Running: {script}")
    try:
        result = subprocess.run(command, check=True, env=env, capture_output=True, text=True)
        print(f"✅ {script} completed successfully.\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error while running {script}:\n--- STDOUT ---\n{e.stdout}\n--- STDERR ---\n{e.stderr}")
        raise

def generate_final_csv():
    """
    Aggregates all agent outputs and generates final_selected_candidates.csv.
    """
    try:
        scores_df = pd.read_csv("cv_grading_results.csv")
        persona_df = pd.read_csv("persona_fit_results.csv")
        bias_df = pd.read_csv("cv_bias_fairness.csv")
        explain_df = pd.read_csv("explainability_results.csv")

        merged = scores_df.copy()

        # Ensure merge safety even if columns are missing
        merged["persona_fit_score"] = persona_df.get("Persona_Score", [0]*len(merged))
        merged["cv_bias_flags"] = bias_df.get("Bias_Flags", ["[]"]*len(merged))
        merged["explanation"] = explain_df.get("Explanation", ["No explanation provided"]*len(merged))

        # Compute final score
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
        print("✅ Final result saved to: final_selected_candidates.csv")

    except Exception as e:
        print(f"❌ Failed to generate final CSV: {e}")
        raise

def main(args):
    agents = [
        "jd_optimizer.py",
        "cv_grader.py",
        "bias_agent.py",
        "persona_agent.py",
        "explainability_agent.py",
        "feedback_agent.py",  # Optional - can be skipped if not implemented
        "sql_agent.py"
    ]

    for script in agents:
        run_agent(script)

    print("\n📊 Aggregating outputs into final CSV...")
    generate_final_csv()

    # Print a quick preview of top candidates
    if os.path.exists(args.final_selected):
        final_df = pd.read_csv(args.final_selected, encoding="utf-8")
        print("\n🎯 Final Top Candidates Preview:")
        print(final_df[["candidate_filename", "updated_score", "grade_score", "persona_fit_score"]].head(10).to_string(index=False))
    else:
        print("⚠️ final_selected_candidates.csv was not generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HireSense | Supervisor Agent for Full Pipeline")
    parser.add_argument("--final_selected", type=str, default="final_selected_candidates.csv",
                        help="Output CSV file with ranked candidates")
    args = parser.parse_args()
    main(args)
