from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO
app=FastAPI(title="DATA AUTOPILOT Analysis API",version="0.1.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
@app.get("/health")
def health(): return {"status":"ok","service":"analysis-api"}
@app.post("/profile")
async def profile(file:UploadFile=File(...)):
    if not file.filename.lower().endswith(".csv"): raise HTTPException(400,"CSV required")
    df=pd.read_csv(BytesIO(await file.read()))
    n=df.select_dtypes(include="number")
    return {"rows":len(df),"columns":len(df.columns),"missing":int(df.isna().sum().sum()),"duplicates":int(df.duplicated().sum()),"numeric_columns":list(n.columns),"describe":n.describe().replace({float("nan"):None}).to_dict()}
