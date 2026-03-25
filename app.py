"""
Supply Chain AI — FastAPI Backend
Replaces the Streamlit app.py
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import io
import os
import importlib.util
import uuid
from typing import Optional
import dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

dotenv.load_dotenv()

app = FastAPI(title="Supply Chain AI", version="1.0.0")

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store  (swap for Redis in production) ───────────────────
_sessions: dict[str, pd.DataFrame] = {}


# ── Dynamic imports ────────────────────────────────────────────────────────────
def import_from_path(module_name, file_path, func_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)

base_dir = os.getenv("MODEL_DIR", "")

run_lstm_demand_forecast = import_from_path(
    "lstm_train", os.path.join(base_dir, "lstm_train.py"), "run_lstm_demand_forecast"
)
run_montecarlo_risk = import_from_path(
    "montecarlo", os.path.join(base_dir, "montecarlo.py"), "run_montecarlo_risk"
)
run_supplier_risk = import_from_path(
    "supplier_risk", os.path.join(base_dir, "supplier_risk.py"), "run_supplier_risk"
)


# ── LLM ───────────────────────────────────────────────────────────────────────
def get_chain() -> LLMChain:
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

_chain: Optional[LLMChain] = None

def chain() -> LLMChain:
    global _chain
    if _chain is None:
        _chain = get_chain()
    return _chain


# ── Agent logic ────────────────────────────────────────────────────────────────
def agent_answer(df: pd.DataFrame, question: str) -> Optional[str]:
    q = question.lower()
    supplier = material = None

    if "Supplier" in df.columns:
        for s in df["Supplier"].dropna().unique():
            if s and str(s).lower() in q:
                supplier = s
                break
    if "Material" in df.columns:
        for m in df["Material"].dropna().unique():
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
            if "Supplier" in df.columns:
                results = results.merge(df[["Material", "Supplier"]], on="Material", how="left")
            if supplier:
                results = results[results["Supplier"].str.lower() == supplier.lower()]
            if material:
                results = results[results["Material"].str.lower() == material.lower()]
            for _, r in results.iterrows():
                out.append(
                    f"Material: {r['Material']}, Supplier: {r.get('Supplier','N/A')}, "
                    f"Demand: {r['Predicted Demand']:.2f}, Reorder: {r['Reorder Qty']:.2f}"
                )
            if not len(results):
                out.append("No matching material or supplier found for demand.")
            for k, v in metrics.items():
                out.append(f"{k}: {v:.2f}")

    if want_risk:
        results, _ = run_lstm_demand_forecast(df)
        if results is None:
            out.append("Forecast failed for risk analysis.")
        else:
            if "Supplier" in df.columns:
                results = results.merge(df[["Material", "Supplier"]], on="Material", how="left")
            if "Lead Time" not in results.columns:
                results["Lead Time"] = 10
            if supplier:
                results = results[results["Supplier"].str.lower() == supplier.lower()]
            if material:
                results = results[results["Material"].str.lower() == material.lower()]
            risk_df = run_montecarlo_risk(results)
            if "Supplier" in results.columns:
                risk_df = risk_df.merge(results[["Material", "Supplier"]], on="Material", how="left")
            for _, r in risk_df.iterrows():
                out.append(
                    f"Material: {r['Material']}, Supplier: {r.get('Supplier','N/A')}, "
                    f"Stockout: {r['Stockout Probability']:.2%}, P95: {r['95% Demand']:.2f}"
                )
            if not len(risk_df):
                out.append("No matching material or supplier found for risk.")

    if want_supplier:
        risk_df, score = run_supplier_risk(df)
        if supplier:
            risk_df = risk_df[risk_df["Supplier"].str.lower() == supplier.lower()]
        if material and "Material" in risk_df.columns:
            risk_df = risk_df[risk_df["Material"].str.lower() == material.lower()]
        for _, r in risk_df.iterrows():
            out.append(
                f"Supplier: {r.get('Supplier','N/A')}, Material: {r.get('Material','N/A')}, "
                f"Risk: {r['Supplier_Risk']}"
            )
        if not len(risk_df):
            out.append("No matching material or supplier found for supplier risk.")
        out.append(f"Score: {score:.2f}")

    if not out:
        return None
    return "\n".join(out)


# ── Pydantic models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    raw_output: Optional[str] = None


class DatasetMeta(BaseModel):
    session_id: str
    filename: str
    rows: int
    columns: int
    unique_suppliers: Optional[int] = None
    unique_materials: Optional[int] = None
    preview: list[dict]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=DatasetMeta)
async def upload_file(file: UploadFile = File(...)):
    """Accept an Excel file; return a session_id and dataset metadata."""
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Only .xlsx files are accepted.")

    contents = await file.read()
    try:
        df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    session_id = str(uuid.uuid4())
    _sessions[session_id] = df

    meta = DatasetMeta(
        session_id=session_id,
        filename=file.filename,
        rows=len(df),
        columns=len(df.columns),
        unique_suppliers=int(df["Supplier"].nunique()) if "Supplier" in df.columns else None,
        unique_materials=int(df["Material"].nunique()) if "Material" in df.columns else None,
        preview=df.head(8).fillna("").to_dict(orient="records"),
    )
    return meta


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Run the agent + LLM and return the answer."""
    df = _sessions.get(req.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")

    raw_output = agent_answer(df, req.question)
    model_output = raw_output if raw_output else "(No model run needed)"

    try:
        answer = chain().run({"question": req.question, "model_output": model_output})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return ChatResponse(answer=answer, raw_output=raw_output)


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Clean up a session."""
    _sessions.pop(session_id, None)
    return {"deleted": session_id}