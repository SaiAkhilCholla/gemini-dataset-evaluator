import json
import os
import pandas as pd
import streamlit as st
from google import genai

st.set_page_config(page_title="Dataset Challenge Evaluator", layout="wide")

LEADERBOARD_FILE = "leaderboard.json"

def load_leaderboard():
    if os.path.exists(LEADERBOARD_FILE):
        try:
            with open(LEADERBOARD_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_leaderboard(lb):
    try:
        with open(LEADERBOARD_FILE, "w") as f:
            json.dump(lb, f)
    except Exception as e:
        st.error(f"Error saving leaderboard: {e}")

if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = load_leaderboard()

# Keeps track of which API key to use next across manual button clicks
if "key_index" not in st.session_state:
    st.session_state.key_index = 0

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Challenge Setup")
    team_leader = st.text_input("Team Leader Name", placeholder="Enter team leader name")
    topic = st.selectbox(
        "Evaluation Category",
        ["Fake News Detection", "Phishing Detection", "Product Review Sentiment", "Cybersecurity Threat", "Emergency Classification"]
    )
    
    st.markdown("---")
    
    # --- SECURED ADMIN PANEL ---
    with st.expander("🔐 Host / Admin Controls"):
        admin_pwd = st.text_input("Admin Password", type="password", placeholder="Enter password")
        expected_pwd = st.secrets.get("admin_password", "srishti2026")
        
        if admin_pwd == expected_pwd:
            if st.button("🔄 Reset Leaderboard"):
                st.session_state.leaderboard = []
                if os.path.exists(LEADERBOARD_FILE):
                    os.remove(LEADERBOARD_FILE)
                st.success("Leaderboard wiped clean!")
                st.rerun()
        elif admin_pwd:
            st.error("Incorrect password.")

# --- MULTI-TAB INTERFACE ---
tab1, tab2 = st.tabs(["📊 Challenge Evaluator", "🏆 Leaderboard"])

# ================= TAB 1: EVALUATOR =================
with tab1:
    st.title("🏆 Dataset Creation Challenge - Global Evaluator")
    uploaded_file = st.file_uploader("Upload Challenge Submission CSV", type=["csv"])

    if uploaded_file and st.button("Evaluate Entire Dataset"):
        if not team_leader.strip():
            st.error("Please enter the Team Leader Name in the sidebar before evaluating!")
        else:
            raw_secrets = st.secrets.get("gemini_api_keys", st.secrets.get("gemini_api_key", ""))
            api_keys_list = [k.strip() for k in raw_secrets.replace(",", "\n").split("\n") if k.strip()]
            
            if not api_keys_list:
                st.error("API keys are missing in Streamlit Secrets. Please configure them.")
            else:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    st.subheader("📊 Dataset Health & Integrity")
                    total_rows = len(df)
                    missing_count = int(df.isnull().sum().sum())
                    dup_count = int(df.duplicated().sum())
                    
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

                    # --- SINGLE SHOT API CALL (NO AUTO-RETRY) ---
                    with st.spinner(f"Evaluating submission for team {team_leader} via Gemini API..."):
                        sample_size = min(10, total_rows)
                        sampled_df = df.sample(sample_size, random_state=42).copy()
                        for col in sampled_df.select_dtypes(include=['object']).columns:
                            sampled_df[col] = sampled_df[col].astype(str).str.slice(0, 120)
                        sample_data = sampled_df.to_string()
                        
                        prompt = f"""
                        You are the Head Judge for a Dataset Creation Challenge focused on '{topic}'.
                        Analyze the following sample of {sample_size} records from a competitor's submitted dataset of {total_rows} total rows:
                        
                        Dataset Sample:
                        {sample_data}
                        
                        Evaluate the entire dataset submission based on:
                        1. Relevance and quality of text content for '{topic}'.
                        2. Label consistency and structure.
                        3. Suitability for training or evaluating machine learning models.
                        
                        Respond strictly in valid JSON format with these exact keys:
                        - "overall_score": an integer score from 0 to 100 representing the total dataset grade.
                        - "verdict": a short summary phrase (e.g., "Critically Flawed", "Excellent Submission").
                        - "strengths": a paragraph outlining what makes this dataset great.
                        - "weaknesses": a paragraph noting any flaws or potential biases.
                        - "recommendation": final feedback for the challenge participant.
                        """
                        
                        # Select the active key for this run, and increment index for the NEXT manual click
                        active_key = api_keys_list[st.session_state.key_index % len(api_keys_list)]
                        st.session_state.key_index = (st.session_state.key_index + 1) % len(api_keys_list)
                        
                        client = genai.Client(api_key=active_key)
                        response = client.models.generate_content(
                            model="gemini-3.6-flash",
                            contents=prompt,
                            config={"temperature": 0.0}
                        )
                        
                        clean_text = response.text.strip()
                        if clean_text.startswith("```json"):
                            clean_text = clean_text[7:-3].strip()
                        elif clean_text.startswith("```"):
                            clean_text = clean_text[3:-3].strip()
                        
                        eval_json = json.loads(clean_text)
                        score = eval_json.get('overall_score', 0)

                    # --- ADD TO LEADERBOARD & SAVE ---
                    entry = {
                        "Team Leader": team_leader,
                        "Dataset File": uploaded_file.name,
                        "Category": topic,
                        "Score": score,
                        "Verdict": eval_json.get('verdict', 'Evaluated')
                    }
                    
                    st.session_state.leaderboard = [item for item in st.session_state.leaderboard if not (item["Team Leader"] == team_leader and item["Dataset File"] == uploaded_file.name)]
                    st.session_state.leaderboard.append(entry)
                    st.session_state.leaderboard = sorted(st.session_state.leaderboard, key=lambda x: x["Score"], reverse=True)
                    save_leaderboard(st.session_state.leaderboard)

                    # --- 3. DISPLAY RESULTS ---
                    st.markdown("---")
                    st.subheader("🏆 Challenge Evaluation Report")
                    
                    score_col, verdict_col = st.columns([1, 2])
                    score_col.metric("Overall Dataset Score", f"{score} / 100")
                    verdict_col.success(f"**Verdict:** {eval_json.get('verdict', 'Evaluated')}")
                    
                    st.markdown(f"**💪 Strengths:**\n{eval_json.get('strengths', 'N/A')}")
                    st.markdown(f"**⚠️ Weaknesses & Biases:**\n{eval_json.get('weaknesses', 'N/A')}")
                    st.markdown(f"**💡 Judge's Recommendation:**\n{eval_json.get('recommendation', 'N/A')}")
                    
                    report_summary = pd.DataFrame([eval_json])
                    st.download_button(
                        "Download Challenge Report Summary",
                        data=report_summary.to_csv(index=False).encode('utf-8'),
                        file_name="challenge_evaluation_report.csv"
                    )
                    
                except Exception as e:
                    # Instantly surfaces real API errors without attempting to mask or retry them
                    st.error(f"Evaluation Error: {e}")

# ================= TAB 2: LEADERBOARD =================
with tab2:
    st.title("🏆 Challenge Leaderboard")
    st.markdown("Top submissions ranked purely by Gemini API evaluation score.")

    if not st.session_state.leaderboard:
        st.info("No submissions evaluated yet. Head over to the **Challenge Evaluator** tab to evaluate a dataset!")
    else:
        lb_df = pd.DataFrame(st.session_state.leaderboard)
        
        trophies = []
        for i in range(len(lb_df)):
            if i == 0:
                trophies.append("🥇 Gold")
            elif i == 1:
                trophies.append("🥈 Silver")
            else:
                trophies.append(f"#{i+1}")
                
        lb_df.insert(0, "Rank", trophies)
        st.dataframe(lb_df, use_container_width=True, hide_index=True)
        
        if len(lb_df) >= 1:
            st.markdown(f"### 🥇 1st Place: **{lb_df.iloc[0]['Team Leader']}** ({lb_df.iloc[0]['Score']} pts)")
        if len(lb_df) >= 2:
            st.markdown(f"### 🥈 2nd Place: **{lb_df.iloc[1]['Team Leader']}** ({lb_df.iloc[1]['Score']} pts)")
