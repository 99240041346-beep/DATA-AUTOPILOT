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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import warnings
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score

app=FastAPI(title="DATA AUTOPILOT Analysis API",version="0.2.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
class TrainRequest(BaseModel):
    rows:list[dict[str,Any]]
    target:str
    task:Optional[str]="auto"
class AutoMLRequest(BaseModel):
    rows:list[dict[str,Any]]
    target:Optional[str]=None
    task:Optional[str]="auto"
    test_size:Optional[float]=0.2
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
def build_preprocessor(X:pd.DataFrame):
    numeric=list(X.select_dtypes(include=np.number).columns)
    categorical=[c for c in X.columns if c not in numeric]
    prep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)])
    return prep,numeric,categorical
def target_candidates(df:pd.DataFrame):
    candidates=[]
    for c in df.columns:
        nunique=df[c].nunique(dropna=True)
        if nunique<=1: continue
        score=0
        if c.lower() in {"target","label","y","outcome","result","price","sales","revenue","churn","class"}: score+=5
        if nunique<=20: score+=2
        if pd.api.types.is_numeric_dtype(df[c]) and nunique>10: score+=1
        if c==df.columns[-1]: score+=2
        candidates.append((score,c))
    return [c for _,c in sorted(candidates,reverse=True)]
def leakage_warnings(df:pd.DataFrame,target:str):
    warnings_list=[]
    y=df[target]
    for c in df.columns:
        if c==target: continue
        if df[c].nunique(dropna=True)<=1: warnings_list.append(f"{c}: constant feature")
        if pd.api.types.is_numeric_dtype(df[c]) and pd.api.types.is_numeric_dtype(y):
            corr=df[[c,target]].corr().iloc[0,1]
            if pd.notna(corr) and abs(corr)>=0.98:
                warnings_list.append(f"{c}: near-perfect correlation with target ({corr:.3f})")
    return warnings_list
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
    df=df.dropna(subset=[req.target])
    if len(df)<8: raise HTTPException(400,"At least 8 usable rows are required for training")
    y=df[req.target];task=infer_task(y,req.task);X=df.drop(columns=[req.target])
    prep,numeric,categorical=build_preprocessor(X)
    if task=="regression":
        model=RandomForestRegressor(n_estimators=180,random_state=42,n_jobs=-1)
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
        pipe=Pipeline([("prep",prep),("model",model)]);pipe.fit(Xtr,ytr);pred=pipe.predict(Xte)
        return {"task":task,"target":req.target,"model":"RandomForestRegressor","metrics":{"r2":round(float(r2_score(yte,pred)),4),"mae":round(float(mean_absolute_error(yte,pred)),4),"rmse":round(float(mean_squared_error(yte,pred)**.5),4)},"baseline":float(np.mean(y)),"rows_used":len(df),"features":X.columns.tolist()}
    if y.nunique()<2: raise HTTPException(400,"Classification target needs at least two classes")
    model=RandomForestClassifier(n_estimators=180,random_state=42,n_jobs=-1,class_weight="balanced")
    strat=y if y.value_counts().min()>=2 else None
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=strat)
    pipe=Pipeline([("prep",prep),("model",model)]);pipe.fit(Xtr,ytr);pred=pipe.predict(Xte)
    return {"task":task,"target":req.target,"model":"RandomForestClassifier","metrics":{"accuracy":round(float(accuracy_score(yte,pred)),4),"f1_weighted":round(float(f1_score(yte,pred,average="weighted")),4)},"classes":[str(x) for x in sorted(y.unique(),key=str)],"rows_used":len(df),"features":X.columns.tolist()}

@app.post("/automl")
def automl(req:AutoMLRequest):
    df=frame(req.rows)
    candidates=target_candidates(df)
    target=req.target or (candidates[0] if candidates else None)
    if not target or target not in df.columns: raise HTTPException(400,"Choose a usable target column")
    df=df.dropna(subset=[target]).copy()
    if len(df)<12: raise HTTPException(400,"At least 12 usable rows are required for AutoPilot experiments")
    y=df[target];task=infer_task(y,req.task);X=df.drop(columns=[target])
    if task=="classification" and y.nunique()<2: raise HTTPException(400,"Classification needs at least two target classes")
    warnings_list=leakage_warnings(df,target)
    prep,numeric,categorical=build_preprocessor(X)
    strat=y if task=="classification" and y.value_counts().min()>=2 else None
    test_size=min(max(float(req.test_size or .2),.1),.4)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=42,stratify=strat)
    if task=="regression":
        models=[("Linear Regression",LinearRegression()),("Random Forest",RandomForestRegressor(n_estimators=180,random_state=42,n_jobs=-1)),("Gradient Boosting",GradientBoostingRegressor(random_state=42))]
    else:
        models=[("Logistic Regression",LogisticRegression(max_iter=1000)),("Random Forest",RandomForestClassifier(n_estimators=180,random_state=42,n_jobs=-1,class_weight="balanced")),("Gradient Boosting",GradientBoostingClassifier(random_state=42))]
    results=[];best_pipe=None;best_score=-np.inf;best_name=None
    for name,model in models:
        try:
            pipe=Pipeline([("prep",prep),("model",model)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore");pipe.fit(Xtr,ytr)
            pred=pipe.predict(Xte)
            if task=="regression":
                metrics={"r2":round(float(r2_score(yte,pred)),4),"mae":round(float(mean_absolute_error(yte,pred)),4),"rmse":round(float(mean_squared_error(yte,pred)**.5),4)}
                score=metrics["r2"]
            else:
                metrics={"accuracy":round(float(accuracy_score(yte,pred)),4),"f1_weighted":round(float(f1_score(yte,pred,average="weighted")),4)}
                score=metrics["f1_weighted"]
            results.append({"model":name,"metrics":metrics,"score":round(float(score),4)})
            if score>best_score: best_score=score;best_name=name;best_pipe=pipe
        except Exception as exc:
            results.append({"model":name,"metrics":{},"error":str(exc)[:180]})
    if best_pipe is None: raise HTTPException(400,"No model could be trained on this dataset")
    importance=[]
    try:
        scoring="r2" if task=="regression" else "f1_weighted"
        perm=permutation_importance(best_pipe,Xte,yte,n_repeats=3,random_state=42,n_jobs=-1,scoring=scoring)
        order=np.argsort(perm.importances_mean)[::-1][:10]
        feature_names=list(X.columns)
        importance=[{"feature":str(feature_names[i]),"importance":round(float(max(perm.importances_mean[i],0)),5)} for i in order if i<len(feature_names) and perm.importances_mean[i]>0]
    except Exception:
        importance=[]
    report=(f"Autopilot detected {task} with '{target}' as the target. It evaluated {len(results)} models and selected {best_name} using the holdout score. "
            + (f"Data-quality warnings: {len(warnings_list)}." if warnings_list else "No high-risk leakage pattern was detected by the baseline checks."))
    return {"task":task,"target":target,"target_candidates":candidates[:8],"rows_used":len(df),"models":results,"best_model":best_name,"best_score":round(float(best_score),4),"feature_importance":importance,"warnings":warnings_list,"report":report}

@app.post("/what-if")
def what_if(req:WhatIfRequest):
    impact=0.0;details={}
    for name,delta in req.changes.items():
        effect=float(req.coefficients.get(name,0.0)*delta);details[name]=round(effect,4);impact+=effect
    return {"baseline":req.baseline,"scenario":round(req.baseline+impact,4),"impact":round(impact,4),"feature_effects":details}
