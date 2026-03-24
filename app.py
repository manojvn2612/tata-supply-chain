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
def import_from_path(module_name, file_path, func_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)

base_dir = os.path.join(os.path.dirname(__file__), 'tata-supply-chain')

run_lstm_demand_forecast = import_from_path(
    'lstm_train',
    os.path.join(base_dir, 'lstm_train.py'),
    'run_lstm_demand_forecast'
)

run_montecarlo_risk = import_from_path(
    'montecarlo',
    os.path.join(base_dir, 'montecarlo.py'),
    'run_montecarlo_risk'
)

run_supplier_risk = import_from_path(
    'supplier_risk',
    os.path.join(base_dir, 'supplier_risk.py'),
    'run_supplier_risk'
)

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="meta-llama/llama-3-8b-instruct",
    temperature=0.2
)

prompt_template = PromptTemplate(
    input_variables=["question", "model_output"],
    template="""
You are an expert supply chain AI agent.

User Question:
{question}

Model Output:
{model_output}

Explain clearly, give reasoning, and suggest action.
"""
)

chain = LLMChain(llm=llm, prompt=prompt_template)

def agent_answer(df, question):
    q = question.lower()

    # Try to extract a supplier or material from the question (case-insensitive substring match)
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

    # Determine which models to run
    want_demand = any(w in q for w in ["demand", "forecast", "reorder", "lstm"])
    want_risk = any(w in q for w in ["monte", "stockout", "simulation", "risk"])
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
                out.append(f"Material: {r['Material']}, Supplier: {r.get('Supplier','N/A')}, Demand: {r['Predicted Demand']:.2f}, Reorder: {r['Reorder Qty']:.2f}")
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
                out.append(f"Material: {r['Material']}, Supplier: {r.get('Supplier','N/A')}, Stockout: {r['Stockout Probability']:.2%}, P95: {r['95% Demand']:.2f}")
            if not len(risk_df):
                out.append("No matching material or supplier found for risk.")

    if want_supplier:
        risk_df, score = run_supplier_risk(df)
        if supplier:
            risk_df = risk_df[risk_df['Supplier'].str.lower() == supplier.lower()]
        if material and 'Material' in risk_df.columns:
            risk_df = risk_df[risk_df['Material'].str.lower() == material.lower()]
        for _, r in risk_df.iterrows():
            out.append(f"Supplier: {r.get('Supplier','N/A')}, Material: {r.get('Material','N/A')}, Risk: {r['Supplier_Risk']}")
        if not len(risk_df):
            out.append("No matching material or supplier found for supplier risk.")
        out.append(f"Score: {score:.2f}")

    if not out:
        return "Specify demand, supplier, or risk"
    return "\n".join(out)

def main():
    st.title("Supply Chain AI Agent")

    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])
    question = st.text_input("Ask question")

    if uploaded_file and question:
        df = pd.read_excel(uploaded_file)

        with st.spinner("Running models..."):
            model_output = agent_answer(df, question)

        with st.spinner("Generating reasoning..."):
            answer = chain.run({
                "question": question,
                "model_output": model_output
            })

        st.subheader("Model Output")
        st.text(model_output)

        st.subheader("AI Explanation")
        st.text(answer)

if __name__ == "__main__":
    main()