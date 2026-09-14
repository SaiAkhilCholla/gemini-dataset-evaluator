import json
import pandas as pd
import streamlit as st
import vertexai
from google.oauth2 import service_account
from vertexai.evaluation import EvalTask, PointwiseMetric, PointwiseMetricPromptTemplate
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Data Science Dataset Evaluator", layout="wide")
st.title("🔬 Data Science Dataset & Model Evaluator")

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
    st.header("⚙️ Evaluation Setup")
    topic = st.selectbox(
        "Classification Task",
        ["Phishing Detection", "Fake News Detection", "Product Review Sentiment", "Cybersecurity Threat", "Emergency Classification"]
    )
    st.info("Project ID and credentials loaded securely.")

def init_gcp():
    if "gcp_service_account" in st.secrets:
        creds_dict = json.loads(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        vertexai.init(project=project_id, location=location, credentials=creds)
    else:
        vertexai.init(project=project_id, location=location)

def get_eval_metric(selected_topic):
    rubrics = {
        "Phishing Detection": {
            "instruction": "Classify text as Phishing or Legitimate.",
            "criteria": {"Phishing": "Urgent requests, credential harvesting, fake links.", "Legitimate": "Normal communication, verifiable sender."},
            "rating_rubric": {"1": "Incorrect classification", "5": "Accurate classification"}
        },
        "Fake News Detection": {
            "instruction": "Classify text as Fake or Real based on its content.",
            "criteria": {"Fake": "Sensationalist, unverified claims, emotional bias.", "Real": "Objective, verified facts, credible sources."},
            "rating_rubric": {"1": "Incorrect classification", "5": "Correct classification"}
        },
        "Product Review Sentiment": {
            "instruction": "Classify review as Positive, Negative, or Neutral.",
            "criteria": {"Positive": "Satisfaction, praise.", "Negative": "Disappointment, defects.", "Neutral": "Indifferent, purely informational."},
            "rating_rubric": {"1": "Incorrect sentiment assignment", "5": "Accurate sentiment assignment"}
        },
        "Cybersecurity Threat": {
            "instruction": "Classify as Malware, Phishing, DDoS, or Other.",
            "criteria": {"Malware": "Payloads/viruses.", "Phishing": "Social engineering.", "DDoS": "Traffic exhaustion.", "Other": "General issues."},
            "rating_rubric": {"1": "Incorrect threat classification", "5": "Correct threat classification"}
        },
        "Emergency Classification": {
            "instruction": "Classify message as Medical, Fire, Flood, Rescue, or Other.",
            "criteria": {"Medical": "Health crises.", "Fire": "Smoke/flames.", "Flood": "Rising water.", "Rescue": "Trapped individuals.", "Other": "Non-urgent."},
            "rating_rubric": {"1": "Incorrect emergency category", "5": "Correct emergency category"}
        }
    }
    cfg = rubrics.get(selected_topic, rubrics["Fake News Detection"])
    return PointwiseMetric(
        metric=f'{selected_topic.lower().replace(" ", "_")}_metric',
        metric_prompt_template=PointwiseMetricPromptTemplate(
            instruction=cfg["instruction"],
            criteria=cfg["criteria"],
            rating_rubric=cfg["rating_rubric"]
        )
    )

# --- MAIN INTERFACE ---
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file and st.button("Run Data Science Evaluation"):
    try:
        df = pd.read_csv(uploaded_file)
        
        # --- DATA SCIENTIST PROFILING ---
        st.subheader("📊 Dataset Health & Profiling")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", len(df))
        col2.metric("Missing Values", df.isnull().sum().sum())
        col3.metric("Duplicate Rows", df.duplicated().sum())
        
        # Class distribution analysis if a label column exists
        label_col = None
        for col in ["reference", "label", "target", "class", "subject"]:
            if col in df.columns:
                label_col = col
                break
                
        if label_col:
            st.write(f"**Distribution based on '{label_col}' column:**")
            st.bar_chart(df[label_col].value_counts())

        # --- STATISTICAL METRICS (SKLEARN) ---
        if "response" in df.columns and "reference" in df.columns:
            st.subheader("📈 Statistical Classification Performance")
            y_true = df["reference"].astype(str)
            y_pred = df["response"].astype(str)
            
            acc = accuracy_score(y_true, y_pred)
            st.metric("Model Accuracy", f"{acc * 100:.2f}%")
            
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            st.write("**Classification Report:**")
            st.dataframe(pd.DataFrame(report).transpose())

        # --- SMART COLUMN MAPPING FOR GEMINI JUDGE ---
        col_mapping = {}
        if "response" not in df.columns:
            if "text" in df.columns:
                col_mapping["response"] = "text"
            elif "title" in df.columns:
                col_mapping["response"] = "title"

        # --- GEMINI LLM JUDGE EVALUATION ---
        with st.spinner("Running Gemini LLM Deep Evaluation..."):
            init_gcp()
            metric = get_eval_metric(topic)
            
            eval_task = EvalTask(
                dataset=df, 
                metrics=[metric],
                metric_column_mapping=col_mapping if col_mapping else None
            )
            result = eval_task.evaluate()
            
            st.subheader("🤖 Gemini Evaluator Insights")
            st.dataframe(result.metrics_table)
            
            st.download_button(
                "Download Complete Analysis CSV",
                data=result.metrics_table.to_csv(index=False).encode('utf-8'),
                file_name="ds_evaluation_results.csv"
            )
    except Exception as e:
        st.error(f"Error executing data analysis: {e}")
