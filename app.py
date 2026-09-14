import json
import pandas as pd
import streamlit as st
import vertexai
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Dataset Challenge Evaluator", layout="wide")
st.title("🏆 Dataset Creation Challenge - Global Evaluator")

# --- BACKGROUND GCP AUTHENTICATION ---
location = "us-central1"
project_id = "dataset-analyser-508617"

try:
    if "gcp_service_account" in st.secrets:
        creds_dict = json.loads(st.secrets["gcp_service_account"])
        project_id = creds_dict.get("project_id", "dataset-analyser-508617")
except Exception:
    pass

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Challenge Criteria")
    topic = st.selectbox(
        "Evaluation Category",
        ["Fake News Detection", "Phishing Detection", "Product Review Sentiment", "Cybersecurity Threat", "Emergency Classification"]
    )
    st.info("Evaluating entire dataset submission quality.")

def init_gcp():
    if "gcp_service_account" in st.secrets:
        creds_dict = json.loads(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        vertexai.init(project=project_id, location=location, credentials=creds)
    else:
        vertexai.init(project=project_id, location=location)

# --- MAIN INTERFACE ---
uploaded_file = st.file_uploader("Upload Challenge Submission CSV", type=["csv"])

if uploaded_file and st.button("Evaluate Entire Dataset"):
    try:
        df = pd.read_csv(uploaded_file)
        
        # --- 1. DATA INTEGRITY & HEALTH METRICS ---
        st.subheader("📊 Dataset Health & Integrity")
        total_rows = len(df)
        missing_count = df.isnull().sum().sum()
        dup_count = df.duplicated().sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Submissions", total_rows)
        col2.metric("Missing Values", missing_count, delta="0 is best" if missing_count == 0 else "Has gaps", delta_color="inverse")
        col3.metric("Duplicate Rate", f"{(dup_count/total_rows)*100:.1f}%")
        
        label_col = None
        for col in ["reference", "label", "target", "class", "subject"]:
            if col in df.columns:
                label_col = col
                break
                
        if label_col:
            class_counts = df[label_col].value_counts()
            col4.metric("Unique Classes", len(class_counts))
            st.write(f"**Class Balance Distribution ({label_col}):**")
            st.bar_chart(class_counts)

        # --- 2. GLOBAL GEMINI DATASET JUDGE ---
        with st.spinner("Gemini is analyzing the entire dataset quality for the challenge..."):
            init_gcp()
            model = GenerativeModel("gemini-1.5-flash")
            
            # Take a smart sample of the dataset for the LLM judge overview
            sample_size = min(15, total_rows)
            sample_data = df.sample(sample_size).to_string()
            
            prompt = f"""
            You are the Head Judge for a Dataset Creation Challenge focused on '{topic}'.
            Analyze the following sample from a competitor's submitted dataset of {total_rows} total rows:
            
            Dataset Sample:
            {sample_data}
            
            Evaluate the entire dataset submission based on:
            1. Relevance and quality of text content for '{topic}'.
            2. Label consistency and structure.
            3. Suitability for training or evaluating machine learning models.
            
            Respond strictly in valid JSON format with these exact keys:
            - "overall_score": an integer score from 0 to 100 representing the total dataset grade.
            - "verdict": a short summary phrase (e.g., "Excellent Submission", "Needs Refinement").
            - "strengths": a paragraph outlining what makes this dataset great.
            - "weaknesses": a paragraph noting any flaws or potential biases.
            - "recommendation": final feedback for the challenge participant.
            """
            
            response = model.generate_content(prompt)
            clean_text = response.text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:-3].strip()
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:-3].strip()
                
            eval_result = json.loads(clean_text)
            
            # --- 3. DISPLAY CHALLENGE RESULTS ---
            st.markdown("---")
            st.subheader("🏆 Challenge Evaluation Report")
            
            score_col, verdict_col = st.columns([1, 2])
            score_col.metric("Overall Dataset Score", f"{eval_result.get('overall_score', 0)} / 100")
            verdict_col.success(f"**Verdict:** {eval_result.get('verdict', 'Evaluated')}")
            
            st.markdown(f"**💪 Strengths:**\n{eval_result.get('strengths', 'N/A')}")
            st.markdown(f"**⚠️ Weaknesses & Biases:**\n{eval_result.get('weaknesses', 'N/A')}")
            st.markdown(f"**💡 Judge's Recommendation:**\n{eval_result.get('recommendation', 'N/A')}")
            
            # Export Report Summary
            report_summary = pd.DataFrame([eval_result])
            st.download_button(
                "Download Challenge Report Summary",
                data=report_summary.to_csv(index=False).encode('utf-8'),
                file_name="challenge_evaluation_report.csv"
            )
            
    except Exception as e:
        st.error(f"Error evaluating dataset: {e}")
