# Supply Chain AI Platform

A modern supply chain analytics platform leveraging AI/ML for demand forecasting, supplier risk analysis, and interactive dashboards.

## Features

- Excel Upload
- Demand Forecasting
- Supplier Risk Analysis
- Interactive Dashboard
- Conversational AI
- Monte Carlo Simulation

## Technologies

### Backend

- Framework = Fast API (Python)

It orchestrates between lstm which is for demand Forecasting , Monte Carlo Simulation which is for simulation of stockout of each item and supplier risk which uses Kmeans which Dynamically allots risks to vendors which helps us to order from low risk vendor and take decision.

Endpoints = for backend
- POST /upload: Upload Excel, returns session and dataset info.
- POST /chat: Ask questions, get AI/ML-powered answers.
- DELETE /session/{id}: Clean up session.
- GET /health: Health check.

### Frontend

- Framework =  React (Vite)
- Features = File upload, chat interface, dashboard with KPIs and charts, CSV export.

## Setup

### Backend
1. Install Python 3.11+ and create a virtual environment.
2. Install dependencies = ```pip install -r requirements.txt```
3. Set environment variables in `.env`
    - OPENROUTER_API_KEY (for LLM features)
    - MODEL_DIR (if models are in a custom path)
    - ALLOWED_ORIGINS (CORS, optional)
4. Run the server = ```uvicorn app:app --host 0.0.0.0 --port 8000```

### Frontend
1. Install Node.js (18+ recommended).
2. In frontend/:
3. Open http://localhost:5173 in your browser.

# Images
![frontend](https://github.com/user-attachments/assets/767a1117-2225-4e3e-a835-eb367e189072)

Made By students of MIT WPU
- [Manoj Nayak](https://github.com/manojvn2612)
- [Shalaka Bhor](https://github.com/catana-11)
- [Pranoti Patil](https://github.com/manojvn2612/tata-supply-chain)

Pls any doubt contact us!!!
