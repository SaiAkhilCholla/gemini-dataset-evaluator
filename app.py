import json
import time
import pandas as pd
import streamlit as st
from google import genai
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Dataset Challenge Evaluator", layout="wide")
st.title("🏆 Dataset Creation Challenge - Global Evaluator")

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Challenge Criteria")
    topic = st.selectbox(
        "Evaluation Category",
        ["Fake News Detection", "Phishing Detection", "Product Review Sentiment", "Cybersecurity Threat", "Emergency Classification"]
    )
    
    api_key = st.secrets.get("gemini_api_key", "")
    if not api_key:
        api_key = st.text_input("Gemini API Key", type="password")
    
    st.info("Connected to Google AI Studio (Free Tier). Auto-retry enabled for high demand.")

# --- ROBUST API CALL WITH RETRY ---
def call_gemini_with_retry(client, model, prompt, max_retries=3):
    delay = 2
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=prompt)
        except Exception as e:
            err_msg = str(e)
            if ("503" in err_msg or "UNAVAILABLE" in err_msg or "overloaded" in err_msg) and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                continue
            raise e

# --- MAIN INTERFACE ---
uploaded_file = st.file_uploader("Upload Challenge Submission CSV", type=["csv"])

if uploaded_file and st.button("Evaluate Entire Dataset"):
    if not api_key:
        st.error("Please provide your Gemini API Key in Streamlit Secrets.")
    else:
        try:
            df = pd.read_csv(uploaded_file)
            
            # --- 1. DATA INTEGRITY & HEALTH METRICS ---
            st.subheader("📊 Dataset Health & Integrity")
            total_rows = len(df)
            missing_count = df.isnull().sum().sum()
            dup_count = df.duplicated().sum()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Submissions", total_rows)
            col2.metric("Missing Values", missing_count)
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
            with st.spinner("Gemini is analyzing the entire dataset quality (retrying automatically if busy)..."):
                client = genai.Client(api_key=api_key)
                
                # Fixed seed for consistent reproducible scoring
                sample_size = min(15, total_rows)
                sample_data = df.sample(sample_size, random_state=42).to_string()
                
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
                # --- Update this call to include temperature=0.0 ---
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt,
                    config={"temperature": 0.0}  # <--- Forces deterministic scoring
                )
                
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
                
                report_summary = pd.DataFrame([eval_result])
                st.download_button(
                    "Download Challenge Report Summary",
                    data=report_summary.to_csv(index=False).encode('utf-8'),
                    file_name="challenge_evaluation_report.csv"
                )
                
        except Exception as e:
            st.error(f"Error evaluating dataset: {e}")
