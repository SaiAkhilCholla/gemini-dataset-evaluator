import json
import pandas as pd
import streamlit as st
import vertexai
from google.oauth2 import service_account
from vertexai.evaluation import EvalTask, PointwiseMetric, PointwiseMetricPromptTemplate

st.set_page_config(page_title="GenAI Dataset Evaluator", layout="wide")
st.title("📊 GenAI Dataset Evaluator")

# --- GCP AUTHENTICATION ---
with st.sidebar:
    st.header("1. GCP Setup")
    project_id = st.text_input("GCP Project ID", placeholder="your-project-id")
    location = st.text_input("Region", value="us-central1")
    
    st.header("2. Evaluation Setup")
    topic = st.selectbox(
        "Classification Task",
        ["Phishing Detection", "Fake News Detection", "Product Review Sentiment", "Cybersecurity Threat", "Emergency Classification"]
    )

# Initialize GCP Auth (Uses Secrets in cloud, default local auth if offline)
def init_gcp():
    if "gcp_service_account" in st.secrets:
        creds_dict = json.loads(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        vertexai.init(project=project_id, location=location, credentials=creds)
    else:
        vertexai.init(project=project_id, location=location)

# --- TOPIC METRICS DEFINITION ---
def get_eval_metric(selected_topic):
    rubrics = {
        "Phishing Detection": {
            "instruction": "Classify text as Phishing or Legitimate.",
            "criteria": {"Phishing": "Urgent requests, credential harvesting, fake links.", "Legitimate": "Normal communication, verifiable sender."},
            "rating_rubric": {"1": "Incorrect classification or failure to follow criteria.", "5": "Accurate classification matching the criteria."}
        },
        "Fake News Detection": {
            "instruction": "Classify text as Fake or Real.",
            "criteria": {"Fake": "Sensationalist, unverified claims, emotional bias.", "Real": "Objective, verified facts, credible sources."},
            "rating_rubric": {"1": "Incorrect classification.", "5": "Correct classification."}
        },
        "Product Review Sentiment": {
            "instruction": "Classify review as Positive, Negative, or Neutral.",
            "criteria": {"Positive": "Satisfaction, praise.", "Negative": "Disappointment, defects.", "Neutral": "Indifferent, purely informational."},
            "rating_rubric": {"1": "Incorrect sentiment assignment.", "5": "Accurate sentiment assignment."}
        },
        "Cybersecurity Threat": {
            "instruction": "Classify as Malware, Phishing, DDoS, or Other.",
            "criteria": {"Malware": "Payloads/viruses.", "Phishing": "Social engineering.", "DDoS": "Traffic exhaustion.", "Other": "General issues."},
            "rating_rubric": {"1": "Incorrect threat classification.", "5": "Correct threat classification."}
        },
        "Emergency Classification": {
            "instruction": "Classify message as Medical, Fire, Flood, Rescue, or Other.",
            "criteria": {"Medical": "Health crises.", "Fire": "Smoke/flames.", "Flood": "Rising water.", "Rescue": "Trapped individuals.", "Other": "Non-urgent."},
            "rating_rubric": {"1": "Incorrect emergency category.", "5": "Correct emergency category."}
        }
    }
    cfg = rubrics.get(selected_topic, rubrics["Phishing Detection"])
    
    return PointwiseMetric(
        metric=f'{selected_topic.lower().replace(" ", "_")}_metric',
        metric_prompt_template=PointwiseMetricPromptTemplate(
            instruction=cfg["instruction"],
            criteria=cfg["criteria"],
            rating_rubric=cfg["rating_rubric"]
        )
    )

# --- MAIN INTERFACE ---
uploaded_file = st.file_uploader("Upload CSV (Must contain 'response' and 'reference' columns)", type=["csv"])

if uploaded_file and st.button("Run Evaluation"):
    if not project_id:
        st.error("Please provide your GCP Project ID in the sidebar.")
    else:
        try:
            init_gcp()
            df = pd.read_csv(uploaded_file)
            
            with st.spinner("Gemini is evaluating your dataset..."):
                metric = get_eval_metric(topic)
                eval_task = EvalTask(dataset=df, metrics=[metric, "exact_match"])
                result = eval_task.evaluate()
                
                st.success("Evaluation Finished!")
                st.metric("Exact Match Accuracy", f"{result.summary_metrics.get('exact_match', 0) * 100:.1f}%")
                st.dataframe(result.metrics_table)
                
                st.download_button(
                    "Download Results CSV",
                    data=result.metrics_table.to_csv(index=False).encode('utf-8'),
                    file_name="evaluation_results.csv"
                )
        except Exception as e:
            st.error(f"Error running evaluation: {e}")
