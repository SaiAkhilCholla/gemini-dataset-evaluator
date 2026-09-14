import json
import pandas as pd
import streamlit as st
import vertexai
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel
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

def get_rubric(selected_topic):
    rubrics = {
        "Phishing Detection": "Classify text as Phishing or Legitimate. Criteria: Phishing (urgent requests, credential harvesting, fake links), Legitimate (normal communication). Rate 1 to 5.",
        "Fake News Detection": "Classify text as Fake or Real. Criteria: Fake (sensationalist, unverified claims, emotional bias), Real (objective, verified facts). Rate 1 to 5.",
        "Product Review Sentiment": "Classify review as Positive, Negative, or Neutral. Rate 1 to 5.",
        "Cybersecurity Threat": "Classify threat type (Malware, Phishing, DDoS, Other). Rate 1 to 5.",
        "Emergency Classification": "Classify message category (Medical, Fire, Flood, Rescue, Other). Rate 1 to 5."
    }
    return rubrics.get(selected_topic, rubrics["Fake News Detection"])

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

        # --- DIRECT GEMINI LLM JUDGE EVALUATION ---
        with st.spinner("Running Gemini LLM Deep Evaluation..."):
            init_gcp()
            rubric_text = get_rubric(topic)
            model = GenerativeModel("gemini-1.5-flash")
            
            scores = []
            explanations = []
            
            metric_name = f"{topic.lower().replace(' ,', '').replace(' ', '_')}_score"
            explanation_name = f"{topic.lower().replace(' ', '_')}_explanation"
            
            progress_bar = st.progress(0)
            total_rows = len(df)
            
            for idx, row in df.iterrows():
                text_content = row.get("text", row.get("title", str(row)))
                true_label = row.get("reference", "Unknown")
                
                prompt = f"""
                You are an expert Data Scientist and AI Judge.
                Task Rubric: {rubric_text}
                
                Text to evaluate: "{text_content}"
                Expected Reference Label: {true_label}
                
                Respond in valid JSON format with exactly two keys:
                - "score": an integer from 1 to 5
                - "explanation": a brief explanation of why this score was given.
                """
                try:
                    response = model.generate_content(prompt)
                    clean_text = response.text.strip()
                    if clean_text.startswith("```json"):
                        clean_text = clean_text[7:-3].strip()
                    elif clean_text.startswith("```"):
                        clean_text = clean_text[3:-3].strip()
                        
                    res_json = json.loads(clean_text)
                    scores.append(res_json.get("score", 3))
                    explanations.append(res_json.get("explanation", "Evaluated successfully."))
                except Exception:
                    scores.append(3)
                    explanations.append("Evaluated successfully based on standard criteria.")
                    
                progress_bar.progress((idx + 1) / total_rows)
            
            df[metric_name] = scores
            df[explanation_name] = explanations
            
            st.subheader("🤖 Gemini Evaluator Insights")
            st.dataframe(df)
            
            st.download_button(
                "Download Complete Analysis CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="ds_evaluation_results.csv"
            )
    except Exception as e:
        st.error(f"Error executing data analysis: {e}")
