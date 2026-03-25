
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("supplychain-backend")
dotenv.load_dotenv()
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MODEL_DIR = os.getenv("MODEL_DIR", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY not set. LLM features will not work.")

app = FastAPI(title="Supply Chain AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, pd.DataFrame] = {}

def import_from_path(module_name, file_path, func_name):
    if not os.path.exists(file_path):
        logger.error(f"Module file not found: {file_path}")
        raise ImportError(f"Module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)

try:
    run_lstm_demand_forecast = import_from_path(
        "lstm_train", os.path.join(MODEL_DIR, "lstm_train.py"), "run_lstm_demand_forecast"
    )
    run_montecarlo_risk = import_from_path(
        "montecarlo", os.path.join(MODEL_DIR, "montecarlo.py"), "run_montecarlo_risk"
    )
    run_supplier_risk = import_from_path(
        "supplier_risk", os.path.join(MODEL_DIR, "supplier_risk.py"), "run_supplier_risk"
    )
except Exception as e:
    logger.error(f"Error importing model modules: {e}")
    raise

def get_chain() -> LLMChain:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set.")
    llm = ChatOpenAI(
        openai_api_key=OPENROUTER_API_KEY,
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
        try:
            results, metrics = run_lstm_demand_forecast(df)
        except Exception as e:
            logger.error(f"LSTM demand forecast failed: {e}")
            out.append(f"LSTM failed: {e}")
            results, metrics = None, {}
        if results is not None:
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
        try:
            results, _ = run_lstm_demand_forecast(df)
        except Exception as e:
            logger.error(f"Forecast failed for risk analysis: {e}")
            out.append(f"Forecast failed for risk analysis: {e}")
            results = None
        if results is not None:
            if "Supplier" in df.columns:
                results = results.merge(df[["Material", "Supplier"]], on="Material", how="left")
            if "Lead Time" not in results.columns:
                results["Lead Time"] = 10
            if supplier:
                results = results[results["Supplier"].str.lower() == supplier.lower()]
            if material:
                results = results[results["Material"].str.lower() == material.lower()]
            try:
                risk_df = run_montecarlo_risk(results)
            except Exception as e:
                logger.error(f"Monte Carlo risk failed: {e}")
                out.append(f"Monte Carlo risk failed: {e}")
                risk_df = pd.DataFrame()
            if "Supplier" in results.columns and not risk_df.empty:
                risk_df = risk_df.merge(results[["Material", "Supplier"]], on="Material", how="left")
            for _, r in risk_df.iterrows():
                out.append(
                    f"Material: {r['Material']}, Supplier: {r.get('Supplier','N/A')}, "
                    f"Stockout: {r['Stockout Probability']:.2%}, P95: {r['95% Demand']:.2f}"
                )
            if risk_df.empty:
                out.append("No matching material or supplier found for risk.")

    if want_supplier:
        try:
            risk_df, score = run_supplier_risk(df)
        except Exception as e:
            logger.error(f"Supplier risk failed: {e}")
            out.append(f"Supplier risk failed: {e}")
            risk_df, score = pd.DataFrame(), 0
        if supplier and not risk_df.empty:
            risk_df = risk_df[risk_df["Supplier"].str.lower() == supplier.lower()]
        if material and "Material" in risk_df.columns and not risk_df.empty:
            risk_df = risk_df[risk_df["Material"].str.lower() == material.lower()]
        for _, r in risk_df.iterrows():
            out.append(
                f"Supplier: {r.get('Supplier','N/A')}, Material: {r.get('Material','N/A')}, "
                f"Risk: {r['Supplier_Risk']}"
            )
        if risk_df.empty:
            out.append("No matching material or supplier found for supplier risk.")
        out.append(f"Score: {score:.2f}")

    if not out:
        return None
    return "\n".join(out)

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

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload", response_model=DatasetMeta)
async def upload_file(file: UploadFile = File(...)):
    """Accept an Excel file; return a session_id and dataset metadata."""
    if not file.filename.endswith(".xlsx"):
        logger.warning(f"Rejected upload: {file.filename}")
        raise HTTPException(status_code=400, detail="Only .xlsx files are accepted.")

    contents = await file.read()
    try:
        df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        logger.error(f"Could not parse file: {e}")
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
    logger.info(f"Uploaded file: {file.filename} (session: {session_id})")
    return meta

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Run the agent + LLM and return the answer."""
    df = _sessions.get(req.session_id)
    if df is None:
        logger.warning(f"Session not found: {req.session_id}")
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")

    raw_output = agent_answer(df, req.question)
    model_output = raw_output if raw_output else "(No model run needed)"

    try:
        answer = chain().run({"question": req.question, "model_output": model_output})
    except Exception as e:
        logger.error(f"LLM error: {e}")
        raise HTTPException(status_code=500, detail="LLM error.")

    return ChatResponse(answer=answer, raw_output=raw_output)

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Clean up a session."""
    _sessions.pop(session_id, None)
    logger.info(f"Session deleted: {session_id}")
    return {"deleted": session_id}