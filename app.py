# app.py
import streamlit as st
import pandas as pd
import os
import shutil
import tempfile
import subprocess
import sys  # ✅ Added to ensure subprocess uses current Python interpreter

class HireSenseDashboard:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.agents_dir = os.path.join(self.base_dir, 'Agents')

    def setup_workspace(self, temp_dir, job_title, job_description, uploaded_files):
        """Prepare the working directory with job description and CVs"""
        try:
            dataset_dir = os.path.join(temp_dir, 'Dataset')
            cv_dir = os.path.join(dataset_dir, 'CVs1')
            os.makedirs(cv_dir, exist_ok=True)

            # Save the job description to CSV
            pd.DataFrame({
                'Job Title': [job_title],
                'Job Description': [job_description]
            }).to_csv(os.path.join(dataset_dir, 'job_description.csv'), index=False)

            # Save uploaded CVs
            for uploaded_file in uploaded_files:
                file_path = os.path.join(cv_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

            # Copy all agents to the temporary directory
            for agent in [
                "jd_optimizer.py",
                "cv_grader.py",
                "bias_agent.py",
                "persona_agent.py",
                "explainability_agent.py",
                "feedback_agent.py",  # Optional
                "sql_agent.py",
                "supervisor.py"
            ]:
                src = os.path.join(self.agents_dir, agent)
                dst = os.path.join(temp_dir, agent)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                else:
                    raise FileNotFoundError(f"Missing agent: {agent}")

            # Copy supporting files if present
            for file in os.listdir(self.agents_dir):
                if file.endswith(".csv") or file.endswith(".db"):
                    shutil.copy2(os.path.join(self.agents_dir, file), os.path.join(temp_dir, file))

        except Exception as e:
            st.error(f"❌ Workspace setup failed: {str(e)}")
            raise

    def process_candidates(self, job_title, job_description, uploaded_files, top_n):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_dir = os.getcwd()
            try:
                self.setup_workspace(temp_dir, job_title, job_description, uploaded_files)
                os.chdir(temp_dir)

                env = os.environ.copy()
                env["TRANSFORMERS_NO_TF"] = "1"

                result = subprocess.run(
                    [sys.executable, "supervisor.py"],  # ✅ Fixed to use the current interpreter
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env
                )

                st.success("✅ Pipeline executed successfully.")
                st.code(result.stdout)

                results_path = os.path.join(temp_dir, 'final_selected_candidates.csv')
                if os.path.exists(results_path):
                    df = pd.read_csv(results_path)

                    top_candidates = df.nlargest(top_n, 'updated_score')
                    output = []
                    for _, row in top_candidates.iterrows():
                        output.append({
                            'candidate': row.get('candidate_id', row.get('candidate_filename')),
                            'match_score': row['updated_score'] * 100,
                            'cv_score': row['grade_score'] * 100,
                            'persona_score': row['persona_fit_score'] * 100,
                            'bias_free_score': (1 - len(eval(row['cv_bias_flags'])) / 10) * 100 if isinstance(row['cv_bias_flags'], str) else 100,
                            'explanation': row['explanation']
                        })

                    # Save results back to agents folder
                    shutil.copy2(results_path, os.path.join(self.agents_dir, 'final_selected_candidates.csv'))
                    os.chdir(original_dir)
                    return output

                else:
                    raise FileNotFoundError("Results file not found.")

            except subprocess.CalledProcessError as e:
                st.error("❌ Pipeline execution failed.")
                st.text(e.stderr)
                os.chdir(original_dir)
                raise
            except Exception as e:
                os.chdir(original_dir)
                raise

def main():
    st.set_page_config(page_title="HireSense Dashboard", page_icon="🤖", layout="wide")
    st.title("🎯 HireSense – AI-Powered Talent Matching")

    st.header("📄 Enter Job Description")
    col1, col2 = st.columns([1, 2])
    job_title = col1.text_input("Job Title")
    job_description = col2.text_area("Job Description", height=150)

    st.header("📤 Upload Candidate CVs (PDF)")
    uploaded_files = st.file_uploader("Upload multiple CVs", type=["pdf"], accept_multiple_files=True)

    st.header("🎛️ Configuration")
    top_n = st.slider("Number of top candidates to display", 1, 20, 5)

    if st.button("🚀 Analyze Candidates"):
        if not job_title or not job_description or not uploaded_files:
            st.warning("⚠️ Please fill in all required fields before processing.")
        else:
            dashboard = HireSenseDashboard()
            with st.spinner("Processing... Please wait."):
                try:
                    results = dashboard.process_candidates(job_title, job_description, uploaded_files, top_n)

                    st.success("🎉 Top candidates identified!")
                    st.header("🏆 Final Shortlist")
                    for i, res in enumerate(results, 1):
                        with st.expander(f"#{i} – {res['candidate']} (Match Score: {res['match_score']:.2f}%)"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**📊 Score Breakdown:**")
                                st.write(f"CV Score: {res['cv_score']:.2f}%")
                                st.write(f"Persona Fit: {res['persona_score']:.2f}%")
                                st.write(f"Bias-Free Score: {res['bias_free_score']:.2f}%")
                            with col2:
                                st.markdown("**💡 Explanation:**")
                                st.write(res['explanation'])

                    st.download_button(
                        "📥 Download Results CSV",
                        pd.DataFrame(results).to_csv(index=False).encode("utf-8"),
                        "hiresense_results.csv",
                        "text/csv"
                    )

                except Exception as e:
                    st.error(f"⚠️ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
