import streamlit as st
import pandas as pd
import os
import importlib.util
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import dotenv

dotenv.load_dotenv()
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain AI",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background: #0d0f14;
    color: #e2e8f0;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #12151c;
    border-right: 1px solid #1e2433;
  }
  section[data-testid="stSidebar"] .stMarkdown h1,
  section[data-testid="stSidebar"] .stMarkdown h2,
  section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #38bdf8;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* File uploader */
  [data-testid="stFileUploader"] {
    background: #1a1f2e;
    border: 1px dashed #2d3748;
    border-radius: 8px;
    padding: 12px;
  }

  /* Main chat area */
  .chat-container {
    max-width: 820px;
    margin: 0 auto;
    padding-bottom: 120px;
  }

  /* User bubble */
  .bubble-user {
    display: flex;
    justify-content: flex-end;
    margin: 12px 0;
  }
  .bubble-user .msg {
    background: #1d4ed8;
    color: #eff6ff;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 70%;
    font-size: 0.92rem;
    line-height: 1.5;
  }

  /* Assistant bubble */
  .bubble-ai {
    display: flex;
    justify-content: flex-start;
    margin: 12px 0;
    gap: 10px;
    align-items: flex-start;
  }
  .bubble-ai .avatar {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .bubble-ai .msg {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    color: #e2e8f0;
    padding: 12px 16px;
    border-radius: 4px 18px 18px 18px;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.6;
  }
  .bubble-ai .msg pre {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 10px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #a5f3fc;
    overflow-x: auto;
    margin: 8px 0 0 0;
  }
  .bubble-ai .section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
  }

  /* Header */
  .top-header {
    text-align: center;
    padding: 40px 0 20px;
    font-family: 'IBM Plex Mono', monospace;
  }
  .top-header h1 {
    font-size: 1.6rem;
    font-weight: 600;
    color: #f8fafc;
    letter-spacing: -0.02em;
  }
  .top-header p {
    color: #64748b;
    font-size: 0.85rem;
  }

  /* Empty state */
  .empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #374151;
  }
  .empty-state .icon { font-size: 3rem; }
  .empty-state p { margin-top: 12px; font-size: 0.9rem; }

  /* Input box override */
  div[data-testid="stChatInput"] {
    background: #12151c !important;
    border-top: 1px solid #1e2433;
    padding: 12px 0;
  }
  div[data-testid="stChatInput"] textarea {
    background: #1a1f2e !important;
    border: 1px solid #2d3748 !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
  }
  div[data-testid="stChatInput"] textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0d0f14; }
  ::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 3px; }

  /* Streamlit default overrides */
  .stButton button {
    background: #1d4ed8;
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.85rem;
    padding: 6px 14px;
  }
  .stButton button:hover { background: #2563eb; }

  /* Metric cards in sidebar */
  .stat-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 12px;
    margin: 6px 0;
  }
  .stat-card .label {
    font-size: 0.72rem;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .stat-card .value {
    font-size: 1.1rem;
    font-weight: 600;
    color: #38bdf8;
    font-family: 'IBM Plex Mono', monospace;
  }
</style>
""", unsafe_allow_html=True)


# ── Dynamic imports ────────────────────────────────────────────────────────────
def import_from_path(module_name, file_path, func_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)

# base_dir = os.path.join(os.path.dirname(__file__), 'tata-supply-chain')
base_dir = ""

run_lstm_demand_forecast = import_from_path(
    'lstm_train', os.path.join(base_dir, 'lstm_train.py'), 'run_lstm_demand_forecast')
run_montecarlo_risk = import_from_path(
    'montecarlo', os.path.join(base_dir, 'montecarlo.py'), 'run_montecarlo_risk')
run_supplier_risk = import_from_path(
    'supplier_risk', os.path.join(base_dir, 'supplier_risk.py'), 'run_supplier_risk')


# ── LLM setup ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_chain():
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="meta-llama/llama-3-8b-instruct",
        temperature=0.2,
    )
    prompt = PromptTemplate(
        input_variables=["question", "model_output"],
        template="""You are an expert supply chain AI agent.

User Question:
{question}

Model Output:
{model_output}

Explain clearly, give reasoning, and suggest action. If you don't know, say so.
If the user asks a theory question, answer from knowledge without running models.""",
    )
    return LLMChain(llm=llm, prompt=prompt)


# ── Agent logic (unchanged from original) ─────────────────────────────────────
def agent_answer(df, question):
    q = question.lower()
    supplier = None
    material = None
    if 'Supplier' in df.columns:
        for s in df['Supplier'].dropna().unique():
            if s and str(s).lower() in q:
                supplier = s
                break
    if 'Material' in df.columns:
        for m in df['Material'].dropna().unique():
            if m and str(m).lower() in q:
                material = m
                break

    want_demand   = any(w in q for w in ["demand", "forecast", "reorder", "lstm"])
    want_risk     = any(w in q for w in ["monte", "stockout", "simulation", "risk"])
    want_supplier = any(w in q for w in ["supplier", "cluster", "kmeans"])

    out = []

    if want_demand:
        results, metrics = run_lstm_demand_forecast(df)
        if results is None:
            out.append(f"LSTM failed: {metrics.get('error', 'error')}")
        else:
            if 'Supplier' in df.columns:
                results = results.merge(df[['Material', 'Supplier']], on='Material', how='left')
            if supplier:
                results = results[results['Supplier'].str.lower() == supplier.lower()]
            if material:
                results = results[results['Material'].str.lower() == material.lower()]
            for _, r in results.iterrows():
                out.append(f"Material: {r['Material']}, Supplier: {r.get('Supplier','N/A')}, "
                           f"Demand: {r['Predicted Demand']:.2f}, Reorder: {r['Reorder Qty']:.2f}")
            if not len(results):
                out.append("No matching material or supplier found for demand.")
            for k, v in metrics.items():
                out.append(f"{k}: {v:.2f}")

    if want_risk:
        results, _ = run_lstm_demand_forecast(df)
        if results is None:
            out.append("Forecast failed for risk analysis.")
        else:
            if 'Supplier' in df.columns:
                results = results.merge(df[['Material', 'Supplier']], on='Material', how='left')
            if "Lead Time" not in results.columns:
                results["Lead Time"] = 10
            if supplier:
                results = results[results['Supplier'].str.lower() == supplier.lower()]
            if material:
                results = results[results['Material'].str.lower() == material.lower()]
            risk_df = run_montecarlo_risk(results)
            if 'Supplier' in results.columns:
                risk_df = risk_df.merge(results[['Material', 'Supplier']], on='Material', how='left')
            for _, r in risk_df.iterrows():
                out.append(f"Material: {r['Material']}, Supplier: {r.get('Supplier','N/A')}, "
                           f"Stockout: {r['Stockout Probability']:.2%}, P95: {r['95% Demand']:.2f}")
            if not len(risk_df):
                out.append("No matching material or supplier found for risk.")

    if want_supplier:
        risk_df, score = run_supplier_risk(df)
        if supplier:
            risk_df = risk_df[risk_df['Supplier'].str.lower() == supplier.lower()]
        if material and 'Material' in risk_df.columns:
            risk_df = risk_df[risk_df['Material'].str.lower() == material.lower()]
        for _, r in risk_df.iterrows():
            out.append(f"Supplier: {r.get('Supplier','N/A')}, Material: {r.get('Material','N/A')}, "
                       f"Risk: {r['Supplier_Risk']}")
        if not len(risk_df):
            out.append("No matching material or supplier found for supplier risk.")
        out.append(f"Score: {score:.2f}")

    if not out:
        return None   # signal: use LLM only
    return "\n".join(out)


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{role, content, raw}]
if "df" not in st.session_state:
    st.session_state.df = None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔗 Supply Chain AI")
    st.markdown("---")

    uploaded = st.file_uploader("Upload supply chain data (.xlsx)", type=["xlsx"])
    if uploaded:
        st.session_state.df = pd.read_excel(uploaded)
        df = st.session_state.df
        st.success(f"Loaded **{uploaded.name}**")

        # Quick stats
        st.markdown("#### Dataset overview")
        st.markdown(f"""
<div class="stat-card">
  <div class="label">Rows</div>
  <div class="value">{len(df):,}</div>
</div>
<div class="stat-card">
  <div class="label">Columns</div>
  <div class="value">{len(df.columns)}</div>
</div>
""", unsafe_allow_html=True)
        if 'Supplier' in df.columns:
            n = df['Supplier'].nunique()
            st.markdown(f"""
<div class="stat-card">
  <div class="label">Unique Suppliers</div>
  <div class="value">{n}</div>
</div>""", unsafe_allow_html=True)
        if 'Material' in df.columns:
            n = df['Material'].nunique()
            st.markdown(f"""
<div class="stat-card">
  <div class="label">Unique Materials</div>
  <div class="value">{n}</div>
</div>""", unsafe_allow_html=True)

        with st.expander("Preview data"):
            st.dataframe(df.head(8), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Suggested questions")
    suggestions = [
        "What is the demand forecast?",
        "Show stockout risk for all materials",
        "Cluster supplier risk",
        "What is safety stock?",
    ]
    for s in suggestions:
        if st.button(s, key=f"sug_{s}"):
            st.session_state._pending_question = s

    st.markdown("---")
    if st.button("🗑 Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
<div style="color:#374151;font-size:0.72rem;margin-top:16px;font-family:'IBM Plex Mono',monospace;">
MODELS: LSTM · Monte Carlo · K-Means<br>
LLM: Llama-3-8B via OpenRouter
</div>""", unsafe_allow_html=True)


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-header">
  <h1>Supply Chain Intelligence</h1>
  <p>Ask about demand forecasts, risk simulations, or supplier clusters</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Render chat history
if not st.session_state.messages:
    st.markdown("""
<div class="empty-state">
  <div class="icon">📦</div>
  <p>Upload your supply chain data and start asking questions.<br>
  I can run LSTM forecasts, Monte Carlo risk simulations, and supplier clustering.</p>
</div>""", unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
<div class="bubble-user">
  <div class="msg">{msg["content"]}</div>
</div>""", unsafe_allow_html=True)
        else:
            raw_block = ""
            if msg.get("raw"):
                raw_block = f"""
<div class="section-label">Model output</div>
<pre>{msg['raw']}</pre>"""
            st.markdown(f"""
<div class="bubble-ai">
  <div class="avatar">🤖</div>
  <div class="msg">
    {raw_block}
    <div class="section-label" style="margin-top:{'8px' if raw_block else '0'}">AI reasoning</div>
    {msg["content"]}
  </div>
</div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── Handle suggestion button clicks ───────────────────────────────────────────
pending = st.session_state.pop("_pending_question", None)

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about demand, risk, or suppliers…") or pending

if user_input:
    if st.session_state.df is None:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ Please upload an Excel file first using the sidebar.",
            "raw": None,
        })
        st.rerun()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    df = st.session_state.df
    chain = get_chain()

    with st.spinner("Running models & generating response…"):
        raw_output = agent_answer(df, user_input)

        if raw_output is None:
            # Theory / general question — skip models
            answer = chain.run({"question": user_input, "model_output": "(No model run needed)"})
        else:
            answer = chain.run({"question": user_input, "model_output": raw_output})

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "raw": raw_output,
    })
    st.rerun()