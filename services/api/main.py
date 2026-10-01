from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score

app=FastAPI(title="DATA AUTOPILOT Analysis API",version="0.2.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
class TrainRequest(BaseModel):
    rows:list[dict[str,Any]]
    target:str
    task:Optional[str]="auto"
class WhatIfRequest(BaseModel):
    baseline:float
    coefficients:dict[str,float]
    changes:dict[str,float]
def frame(rows:list[dict[str,Any]]):
    if not rows: raise HTTPException(400,"No rows supplied")
    return pd.DataFrame(rows)
def infer_task(y:pd.Series,requested:str|None):
    if requested and requested!="auto": return requested
    if pd.api.types.is_numeric_dtype(y) and y.nunique()>10: return "regression"
    return "classification"
@app.get("/health")
def health(): return {"status":"ok","service":"analysis-api","version":"0.2.0"}
@app.post("/profile")
async def profile(file:UploadFile=File(...)):
    if not file.filename.lower().endswith(".csv"): raise HTTPException(400,"CSV required")
    df=pd.read_csv(BytesIO(await file.read()))
    return {"rows":len(df),"columns":len(df.columns),"missing":int(df.isna().sum().sum()),"duplicates":int(df.duplicated().sum()),"numeric_columns":list(df.select_dtypes(include="number").columns),"describe":df.select_dtypes(include="number").describe().replace({np.nan:None}).to_dict()}
@app.post("/train")
def train(req:TrainRequest):
    df=frame(req.rows)
    if req.target not in df.columns: raise HTTPException(400,"Target column not found")
    df=df.dropna(subset=[req.target]);y=df[req.target];task=infer_task(y,req.task);X=df.drop(columns=[req.target])
    numeric=list(X.select_dtypes(include=np.number).columns);categorical=[c for c in X.columns if c not in numeric]
    prep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)])
    if task=="regression":
        model=RandomForestRegressor(n_estimators=180,random_state=42,n_jobs=-1);Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
        pipe=Pipeline([("prep",prep),("model",model)]);pipe.fit(Xtr,ytr);pred=pipe.predict(Xte)
        return {"task":task,"target":req.target,"model":"RandomForestRegressor","metrics":{"r2":round(float(r2_score(yte,pred)),4),"mae":round(float(mean_absolute_error(yte,pred)),4)},"baseline":float(np.mean(y)),"rows_used":len(df),"features":X.columns.tolist()}
    if y.nunique()<2: raise HTTPException(400,"Classification target needs at least two classes")
    model=RandomForestClassifier(n_estimators=180,random_state=42,n_jobs=-1,class_weight="balanced");Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    pipe=Pipeline([("prep",prep),("model",model)]);pipe.fit(Xtr,ytr);pred=pipe.predict(Xte)
    return {"task":task,"target":req.target,"model":"RandomForestClassifier","metrics":{"accuracy":round(float(accuracy_score(yte,pred)),4),"f1_weighted":round(float(f1_score(yte,pred,average="weighted")),4)},"classes":[str(x) for x in sorted(y.unique())],"rows_used":len(df),"features":X.columns.tolist()}
@app.post("/what-if")
def what_if(req:WhatIfRequest):
    impact=0.0;details={}
    for name,delta in req.changes.items():
        effect=float(req.coefficients.get(name,0.0)*delta);details[name]=round(effect,4);impact+=effect
    return {"baseline":req.baseline,"scenario":round(req.baseline+impact,4),"impact":round(impact,4),"feature_effects":details}
