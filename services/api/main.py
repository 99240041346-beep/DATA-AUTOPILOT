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
from sklearn.ensemble import IsolationForest
import warnings
import sqlite3
import json
import os
from datetime import datetime, timezone
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score

app=FastAPI(title="DATA AUTOPILOT Analysis API",version="0.7.0")
DB_PATH=os.getenv("AUTOPILOT_DB_PATH","autopilot.db")
def init_db():
    con=sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, name TEXT NOT NULL,
        rows_used INTEGER, columns_used INTEGER, target TEXT, task TEXT, best_model TEXT,
        best_score REAL, dataset_signature TEXT, report TEXT, payload TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS dataset_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, name TEXT NOT NULL,
        signature TEXT NOT NULL, rows_used INTEGER, columns_used INTEGER,
        columns_json TEXT NOT NULL, payload TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS model_registry (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, name TEXT NOT NULL,
        dataset_signature TEXT NOT NULL, rows_used INTEGER, target TEXT, task TEXT,
        model_name TEXT NOT NULL, score REAL, metrics_json TEXT NOT NULL,
        feature_importance_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'candidate',
        payload TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS monitoring_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
        baseline_version_id INTEGER, baseline_signature TEXT, current_signature TEXT,
        baseline_rows INTEGER, current_rows INTEGER, threshold REAL,
        status TEXT NOT NULL, alerts INTEGER NOT NULL, report TEXT NOT NULL,
        features_json TEXT NOT NULL)""")
    con.commit();con.close()
init_db()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
class TrainRequest(BaseModel):
    rows:list[dict[str,Any]]
    target:str
    task:Optional[str]="auto"
class AnomalyRequest(BaseModel):
    rows:list[dict[str,Any]]
    contamination:Optional[float]=0.05
class ForecastRequest(BaseModel):
    rows:list[dict[str,Any]]
    date_column:str
    value_column:str
    periods:Optional[int]=7
class CleanRequest(BaseModel):
    rows:list[dict[str,Any]]
    target:Optional[str]=None
    z_threshold:Optional[float]=3.0
class CleanResponse(BaseModel):
    rows:list[dict[str,Any]]
    summary:dict[str,Any]
class AutoMLRequest(BaseModel):
    rows:list[dict[str,Any]]
    target:Optional[str]=None
    task:Optional[str]="auto"
    test_size:Optional[float]=0.2
class WhatIfRequest(BaseModel):
    rows:list[dict[str,Any]]
    target:Optional[str]=None
    task:Optional[str]="auto"
    changes:dict[str,Any]
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
class ExperimentRequest(BaseModel):
    name:Optional[str]="Autopilot experiment"
    rows:list[dict[str,Any]]
    result:dict[str,Any]

class DatasetVersionRequest(BaseModel):
    name:Optional[str]="Dataset version"
    rows:list[dict[str,Any]]

class AskRequest(BaseModel):
    rows:list[dict[str,Any]]
    question:str

def dataset_signature(rows:list[dict[str,Any]]):
    raw=json.dumps(rows,sort_keys=True,default=str,separators=(",",":"))
    import hashlib
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

@app.post("/experiments")
def save_experiment(req:ExperimentRequest):
    result=req.result
    con=sqlite3.connect(DB_PATH)
    cur=con.execute("INSERT INTO experiments(created_at,name,rows_used,columns_used,target,task,best_model,best_score,dataset_signature,report,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(),req.name or "Autopilot experiment",len(req.rows),len(req.rows[0]) if req.rows else 0,
         result.get("target"),result.get("task"),result.get("best_model"),result.get("best_score"),dataset_signature(req.rows),
         result.get("report",""),json.dumps(result,default=str)))
    con.commit();eid=cur.lastrowid;con.close()
    return {"id":eid,"saved":True}

@app.get("/experiments")
def list_experiments():
    con=sqlite3.connect(DB_PATH);con.row_factory=sqlite3.Row
    rows=[dict(x) for x in con.execute("SELECT id,created_at,name,rows_used,columns_used,target,task,best_model,best_score,dataset_signature,report FROM experiments ORDER BY id DESC LIMIT 50")]
    con.close();return {"experiments":rows}

@app.get("/experiments/compare")
def compare_experiments(ids:str):
    try:
        wanted=[int(x.strip()) for x in ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400,"Experiment ids must be comma-separated integers")
    if not wanted or len(wanted)>10: raise HTTPException(400,"Provide 1 to 10 experiment ids")
    placeholders=",".join("?" for _ in wanted)
    con=sqlite3.connect(DB_PATH);con.row_factory=sqlite3.Row
    rows=[dict(x) for x in con.execute(f"SELECT id,created_at,name,rows_used,columns_used,target,task,best_model,best_score,dataset_signature FROM experiments WHERE id IN ({placeholders}) ORDER BY id DESC",wanted)]
    con.close()
    return {"experiments":rows,"count":len(rows)}

@app.post("/ask")
def ask(req:AskRequest):
    df=frame(req.rows)
    q=req.question.strip().lower()
    if not q: raise HTTPException(400,"Ask a data question")
    numeric=df.select_dtypes(include=np.number).columns.tolist()
    missing=int(df.isna().sum().sum())
    if any(x in q for x in ["how many rows","number of rows","row count","rows"]):
        answer=f"The dataset contains {len(df):,} rows across {len(df.columns):,} columns."
        return {"answer":answer,"intent":"shape","insights":[{"label":"Rows","value":len(df)},{"label":"Columns","value":len(df.columns)}]}
    if "missing" in q or "null" in q or "empty" in q:
        top=df.isna().sum().sort_values(ascending=False)
        items=[{"feature":str(k),"missing":int(v)} for k,v in top.items() if v>0][:10]
        answer=f"There are {missing:,} missing cells in total."
        return {"answer":answer,"intent":"missingness","insights":items}
    if any(x in q for x in ["average","mean","avg"]):
        matches=[c for c in numeric if c.lower() in q]
        cols=matches or numeric[:5]
        vals=[{"feature":c,"mean":round(float(df[c].mean()),4)} for c in cols if df[c].notna().any()]
        answer="Average values: "+"; ".join(f"{x['feature']} = {x['mean']}" for x in vals)+"."
        return {"answer":answer,"intent":"mean","insights":vals}
    if any(x in q for x in ["highest","maximum","max","largest","top"]):
        vals=[{"feature":c,"max":round(float(df[c].max()),4)} for c in numeric if df[c].notna().any()]
        vals=sorted(vals,key=lambda x:x["max"],reverse=True)[:8]
        answer="Highest observed numeric values: "+"; ".join(f"{x['feature']} = {x['max']}" for x in vals)+"."
        return {"answer":answer,"intent":"max","insights":vals}
    if any(x in q for x in ["lowest","minimum","min","smallest"]):
        vals=[{"feature":c,"min":round(float(df[c].min()),4)} for c in numeric if df[c].notna().any()]
        vals=sorted(vals,key=lambda x:x["min"])[:8]
        answer="Lowest observed numeric values: "+"; ".join(f"{x['feature']} = {x['min']}" for x in vals)+"."
        return {"answer":answer,"intent":"min","insights":vals}
    if "correlation" in q or "correlated" in q or "relationship" in q:
        if len(numeric)<2: raise HTTPException(400,"At least two numeric columns are required for correlation analysis")
        corr=df[numeric].corr()
        pairs=[]
        for i,a in enumerate(numeric):
            for b in numeric[i+1:]:
                v=corr.loc[a,b]
                if pd.notna(v): pairs.append({"feature_a":a,"feature_b":b,"correlation":round(float(v),4)})
        pairs=sorted(pairs,key=lambda x:abs(x["correlation"]),reverse=True)[:8]
        answer="Strongest numeric relationships: "+"; ".join(f"{x['feature_a']} ↔ {x['feature_b']} ({x['correlation']})" for x in pairs)+"."
        return {"answer":answer,"intent":"correlation","insights":pairs}
    if "column" in q or "feature" in q:
        return {"answer":"Columns: "+", ".join(map(str,df.columns))+".","intent":"columns","insights":[{"feature":str(x)} for x in df.columns]}
    return {"answer":f"I analyzed {len(df):,} rows and {len(df.columns):,} columns. Try asking about rows, missing values, averages, highest/lowest values, correlations, or specific columns.","intent":"help","insights":[{"numeric_columns":len(numeric)},{"missing_cells":missing}]}

@app.post("/datasets/versions")
def save_dataset_version(req:DatasetVersionRequest):
    rows=req.rows
    if not rows: raise HTTPException(400,"No rows supplied")
    columns=list(rows[0].keys())
    signature=dataset_signature(rows)
    con=sqlite3.connect(DB_PATH)
    cur=con.execute("INSERT INTO dataset_versions(created_at,name,signature,rows_used,columns_used,columns_json,payload) VALUES(?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(),req.name or "Dataset version",signature,len(rows),len(columns),json.dumps(columns),json.dumps(rows,default=str)))
    con.commit();vid=cur.lastrowid;con.close()
    return {"id":vid,"saved":True,"signature":signature,"rows_used":len(rows),"columns_used":len(columns)}

@app.get("/datasets/versions")
def list_dataset_versions():
    con=sqlite3.connect(DB_PATH);con.row_factory=sqlite3.Row
    rows=[dict(x) for x in con.execute("SELECT id,created_at,name,signature,rows_used,columns_used,columns_json FROM dataset_versions ORDER BY id DESC LIMIT 50")]
    con.close()
    for x in rows: x["columns"]=json.loads(x.pop("columns_json"))
    return {"versions":rows}

@app.get("/datasets/versions/{version_id}")
def get_dataset_version(version_id:int):
    con=sqlite3.connect(DB_PATH);con.row_factory=sqlite3.Row
    row=con.execute("SELECT * FROM dataset_versions WHERE id=?",(version_id,)).fetchone();con.close()
    if not row: raise HTTPException(404,"Dataset version not found")
    item=dict(row);item["columns"]=json.loads(item.pop("columns_json"));item["rows"]=json.loads(item.pop("payload"));return item

class DriftVersionRequest(BaseModel):
    baseline_version_id:int
    current_rows:list[dict[str,Any]]
    threshold:Optional[float]=0.2

@app.post("/drift/version")
def drift_version(req:DriftVersionRequest):
    con=sqlite3.connect(DB_PATH);row=con.execute("SELECT payload FROM dataset_versions WHERE id=?",(req.baseline_version_id,)).fetchone();con.close()
    if not row: raise HTTPException(404,"Baseline dataset version not found")
    baseline=json.loads(row[0])
    return drift(DriftRequest(baseline_rows=baseline,current_rows=req.current_rows,threshold=req.threshold,baseline_version_id=req.baseline_version_id))

@app.get("/experiments/{experiment_id}")
def get_experiment(experiment_id:int):
    con=sqlite3.connect(DB_PATH);con.row_factory=sqlite3.Row
    row=con.execute("SELECT * FROM experiments WHERE id=?",(experiment_id,)).fetchone();con.close()
    if not row: raise HTTPException(404,"Experiment not found")
    item=dict(row);item["payload"]=json.loads(item["payload"]);return item

@app.get("/monitoring/history")
def monitoring_history():
    con=sqlite3.connect(DB_PATH);con.row_factory=sqlite3.Row
    rows=[dict(x) for x in con.execute("SELECT id,created_at,baseline_version_id,baseline_signature,current_signature,baseline_rows,current_rows,threshold,status,alerts,report FROM monitoring_runs ORDER BY id DESC LIMIT 50")]
    con.close()
    return {"runs":rows}

@app.get("/health")
def health(): return {"status":"ok","service":"analysis-api","version":"0.6.0"}
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

@app.post("/anomalies")
def anomalies(req:AnomalyRequest):
    df=frame(req.rows)
    numeric=df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric)<1: raise HTTPException(400,"At least one numeric column is required for anomaly detection")
    X=df[numeric].replace([np.inf,-np.inf],np.nan).fillna(df[numeric].median())
    if len(X)<10: raise HTTPException(400,"At least 10 rows are required for anomaly detection")
    contamination=min(max(float(req.contamination or .05),.01),.25)
    model=IsolationForest(n_estimators=200,contamination=contamination,random_state=42)
    labels=model.fit_predict(X); scores=-model.decision_function(X)
    result=df.copy();result["anomaly"]=labels==-1;result["anomaly_score"]=np.round(scores,5)
    flagged=result[result["anomaly"]].sort_values("anomaly_score",ascending=False).head(100)
    return {"rows_analyzed":len(df),"numeric_features":numeric,"anomalies_found":int((labels==-1).sum()),"anomaly_rate":round(float((labels==-1).mean()),4),"records":flagged.replace({np.nan:None}).to_dict(orient="records")}

@app.post("/forecast")
def forecast(req:ForecastRequest):
    df=frame(req.rows)
    if req.date_column not in df.columns or req.value_column not in df.columns: raise HTTPException(400,"Date or value column not found")
    series=pd.DataFrame({"date":pd.to_datetime(df[req.date_column],errors="coerce"),"value":pd.to_numeric(df[req.value_column],errors="coerce")}).dropna().sort_values("date")
    if len(series)<10: raise HTTPException(400,"At least 10 valid time-series observations are required")
    grouped=series.groupby("date",as_index=False)["value"].mean()
    values=grouped["value"].to_numpy(dtype=float)
    periods=min(max(int(req.periods or 7),1),90)
    window=min(14,len(values))
    recent=values[-window:]
    x=np.arange(window);coef=np.polyfit(x,recent,1) if window>=2 else np.array([0,recent.mean()])
    slope,intercept=float(coef[0]),float(coef[1])
    last_date=grouped["date"].iloc[-1]
    deltas=grouped["date"].diff().dropna().dt.total_seconds()/86400
    step=float(deltas.median()) if len(deltas) else 1.0
    step=max(step,.0001)
    predictions=[]
    for i in range(1,periods+1):
        predictions.append({"date":(last_date+pd.to_timedelta(step*i,unit="D")).isoformat(),"forecast":round(float(slope*(window-1+i)+intercept),4)})
    direction="rising" if slope>0 else "falling" if slope<0 else "stable"
    return {"date_column":req.date_column,"value_column":req.value_column,"observations":len(grouped),"frequency_days":round(step,3),"trend":direction,"trend_per_step":round(slope,5),"forecast":predictions}

@app.post("/clean",response_model=CleanResponse)
def clean_data(req:CleanRequest):
    df=frame(req.rows)
    original_rows=len(df); original_cols=len(df.columns)
    duplicates=int(df.duplicated().sum())
    missing_before={str(k):int(v) for k,v in df.isna().sum().items() if v>0}
    numeric=list(df.select_dtypes(include=np.number).columns)
    outlier_counts={}
    for col in numeric:
        s=df[col].dropna()
        if len(s)>=8 and float(s.std() or 0)>0:
            z=((s-s.mean())/s.std()).abs()
            outlier_counts[col]=int((z>float(req.z_threshold or 3)).sum())
    cleaned=df.drop_duplicates().copy()
    for col in numeric:
        if cleaned[col].isna().any():
            cleaned[col]=cleaned[col].fillna(cleaned[col].median())
    for col in cleaned.columns:
        if col not in numeric and cleaned[col].isna().any():
            mode=cleaned[col].mode(dropna=True)
            if len(mode): cleaned[col]=cleaned[col].fillna(mode.iloc[0])
    missing_after={str(k):int(v) for k,v in cleaned.isna().sum().items() if v>0}
    correlations=[]
    if len(numeric)>=2:
        corr=cleaned[numeric].corr()
        for i,a in enumerate(numeric):
            for b in numeric[i+1:]:
                value=float(corr.loc[a,b])
                if np.isfinite(value) and abs(value)>=.7:
                    correlations.append({"feature_a":a,"feature_b":b,"correlation":round(value,3)})
    summary={
        "original_rows":original_rows,"cleaned_rows":len(cleaned),
        "original_columns":original_cols,"duplicates_removed":duplicates,
        "missing_before":missing_before,"missing_after":missing_after,
        "outliers_by_column":outlier_counts,
        "strong_correlations":sorted(correlations,key=lambda x:abs(x["correlation"]),reverse=True)[:20],
        "numeric_features":numeric,
        "categorical_features":[str(x) for x in cleaned.columns if x not in numeric],
        "cleaning_actions":["removed exact duplicate rows","filled numeric missing values with median","filled categorical missing values with mode"]
    }
    return {"rows":cleaned.replace({np.nan:None}).to_dict(orient="records"),"summary":summary}

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

class ModelRegisterRequest(BaseModel):
    name:Optional[str]="Autopilot model"
    rows:list[dict[str,Any]]
    result:dict[str,Any]

@app.post("/models")
def register_model(req:ModelRegisterRequest):
    result=req.result
    if not result.get("best_model"): raise HTTPException(400,"A completed AutoML result is required")
    sig=dataset_signature(req.rows)
    con=sqlite3.connect(DB_PATH)
    cur=con.execute("INSERT INTO model_registry(created_at,name,dataset_signature,rows_used,target,task,model_name,score,metrics_json,feature_importance_json,status,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(),req.name or "Autopilot model",sig,len(req.rows),result.get("target"),result.get("task"),result["best_model"],result.get("best_score"),json.dumps(next((m.get("metrics",{}) for m in result.get("models",[]) if m.get("model")==result["best_model"]),{})),json.dumps(result.get("feature_importance",[])),"candidate",json.dumps(result)))
    con.commit(); mid=cur.lastrowid;con.close()
    return {"id":mid,"status":"candidate","dataset_signature":sig,"model_name":result["best_model"],"score":result.get("best_score")}

@app.get("/models")
def list_models():
    con=sqlite3.connect(DB_PATH); con.row_factory=sqlite3.Row
    rows=[dict(x) for x in con.execute("SELECT id,created_at,name,dataset_signature,rows_used,target,task,model_name,score,status FROM model_registry ORDER BY id DESC LIMIT 50").fetchall()]
    con.close(); return {"models":rows}

@app.post("/models/{model_id}/promote")
def promote_model(model_id:int):
    con=sqlite3.connect(DB_PATH); row=con.execute("SELECT id FROM model_registry WHERE id=?",(model_id,)).fetchone()
    if not row: con.close(); raise HTTPException(404,"Model version not found")
    con.execute("UPDATE model_registry SET status='candidate' WHERE status='production'")
    con.execute("UPDATE model_registry SET status='production' WHERE id=?",(model_id,))
    con.commit();con.close(); return {"id":model_id,"status":"production"}


@app.post("/explain")
def explain(req:AutoMLRequest):
    df=frame(req.rows)
    candidates=target_candidates(df); target=req.target or (candidates[0] if candidates else None)
    if not target or target not in df.columns: raise HTTPException(400,"Choose a usable target column")
    df=df.dropna(subset=[target]).copy()
    if len(df)<12: raise HTTPException(400,"At least 12 usable rows are required for explainability")
    y=df[target]; task=infer_task(y,req.task); X=df.drop(columns=[target])
    prep,numeric,categorical=build_preprocessor(X)
    if task=="regression": model=RandomForestRegressor(n_estimators=180,random_state=42,n_jobs=-1)
    else: model=RandomForestClassifier(n_estimators=180,random_state=42,n_jobs=-1,class_weight="balanced")
    pipe=Pipeline([("prep",prep),("model",model)])
    pipe.fit(X,y)
    scoring="r2" if task=="regression" else "f1_weighted"
    Xeval=X.tail(min(80,len(X))); yeval=y.loc[Xeval.index]
    try:
        perm=permutation_importance(pipe,Xeval,yeval,n_repeats=3,random_state=42,n_jobs=-1,scoring=scoring)
        order=np.argsort(perm.importances_mean)[::-1][:8]
        global_drivers=[{"feature":str(X.columns[i]),"importance":round(float(max(perm.importances_mean[i],0)),5)} for i in order if i<len(X.columns)]
    except Exception: global_drivers=[]
    row=X.iloc[-1:].copy(); pred=pipe.predict(row)[0]
    local=[]
    base_pred=float(pred) if isinstance(pred,(int,float,np.integer,np.floating)) else str(pred)
    for col in X.columns:
        altered=row.copy()
        if pd.api.types.is_numeric_dtype(X[col]):
            altered[col]=X[col].median()
        else:
            mode=X[col].mode(dropna=True)
            if len(mode): altered[col]=mode.iloc[0]
        try:
            alt=pipe.predict(altered)[0]
            if task=="regression":
                effect=float(pred)-float(alt)
                local.append({"feature":str(col),"effect":round(effect,5)})
            else:
                changed=str(alt)!=str(pred)
                local.append({"feature":str(col),"effect":1 if changed else 0})
        except Exception: pass
    local=sorted(local,key=lambda x:abs(float(x["effect"])),reverse=True)[:8]
    return {"target":target,"task":task,"prediction":base_pred,"row_index":int(df.index[-1]),"global_drivers":global_drivers,"local_drivers":local,"explanation":f"The selected model predicts {base_pred} for the latest usable record. The driver list shows which source features have the strongest measured influence in the evaluation sample."}

@app.post("/what-if")
def what_if(req:WhatIfRequest):
    df=frame(req.rows)
    candidates=target_candidates(df);target=req.target or (candidates[0] if candidates else None)
    if not target or target not in df.columns: raise HTTPException(400,"Choose a usable target column")
    df=df.dropna(subset=[target]).copy()
    if len(df)<12: raise HTTPException(400,"At least 12 usable rows are required for simulation")
    y=df[target];task=infer_task(y,req.task);X=df.drop(columns=[target])
    prep,_,_=build_preprocessor(X)
    model=RandomForestRegressor(n_estimators=180,random_state=42,n_jobs=-1) if task=="regression" else RandomForestClassifier(n_estimators=180,random_state=42,n_jobs=-1,class_weight="balanced")
    pipe=Pipeline([("prep",prep),("model",model)]);pipe.fit(X,y)
    baseline=X.iloc[-1:].copy();scenario=baseline.copy()
    applied=[]
    for name,value in req.changes.items():
        if name not in scenario.columns: continue
        if value is None or value=="": continue
        try:
            scenario.at[scenario.index[0],name]=float(value) if pd.api.types.is_numeric_dtype(X[name]) else value
            applied.append(name)
        except Exception: pass
    base_pred=pipe.predict(baseline)[0];scenario_pred=pipe.predict(scenario)[0]
    if task=="regression":
        impact=float(scenario_pred)-float(base_pred)
        baseline_out=float(base_pred);scenario_out=float(scenario_pred)
    else:
        baseline_out=str(base_pred);scenario_out=str(scenario_pred);impact=0.0 if baseline_out==scenario_out else 1.0
    effects={}
    for name in applied:
        one=scenario.copy()
        one.at[one.index[0],name]=baseline.iloc[0][name]
        p=pipe.predict(one)[0]
        effects[name]=round(float(scenario_pred)-float(p),5) if task=="regression" else (1 if str(p)!=str(scenario_pred) else 0)
    return {"target":target,"task":task,"baseline":baseline_out,"scenario":scenario_out,"impact":round(impact,5),"changed_features":applied,"feature_effects":effects,"message":"Scenario prediction was generated by retraining a fresh baseline model on the supplied dataset."}


class AutopilotRequest(BaseModel):
    rows:list[dict[str,Any]]
    target:Optional[str]=None
    task:Optional[str]="auto"
    date_column:Optional[str]=None
    value_column:Optional[str]=None
    forecast_periods:Optional[int]=7
    run_anomalies:Optional[bool]=True
    run_forecast:Optional[bool]=True

@app.post("/autopilot")
def autopilot(req:AutopilotRequest):
    df=frame(req.rows)
    stages=[]
    started=datetime.now(timezone.utc).isoformat()

    stages.append({"id":"profile","label":"Profile","status":"completed","details":f"{len(df)} rows · {len(df.columns)} columns"})
    cleaned=clean_data(CleanRequest(rows=req.rows,target=req.target))
    clean_rows=cleaned.rows
    clean_summary=cleaned.summary
    stages.append({"id":"clean","label":"Clean","status":"completed","details":f"{clean_summary['duplicates_removed']} duplicates removed · {sum(clean_summary['missing_after'].values())} missing remaining"})

    auto=automl(AutoMLRequest(rows=clean_rows,target=req.target,task=req.task))
    stages.append({"id":"automl","label":"AutoML","status":"completed","details":f"{auto['best_model']} · score {auto['best_score']}"})

    anomaly_result=None
    if req.run_anomalies and len(clean_rows)>=10:
        try:
            anomaly_result=anomalies(AnomalyRequest(rows=clean_rows))
            stages.append({"id":"anomaly","label":"Anomaly","status":"completed","details":f"{anomaly_result['anomalies_found']} anomalies · {round(anomaly_result['anomaly_rate']*100,1)}%"})
        except HTTPException as exc:
            stages.append({"id":"anomaly","label":"Anomaly","status":"skipped","details":str(exc.detail)})
    else:
        stages.append({"id":"anomaly","label":"Anomaly","status":"skipped","details":"Not enough rows or disabled"})

    forecast_result=None
    if req.run_forecast and req.date_column and req.value_column:
        try:
            forecast_result=forecast(ForecastRequest(rows=clean_rows,date_column=req.date_column,value_column=req.value_column,periods=req.forecast_periods or 7))
            stages.append({"id":"forecast","label":"Forecast","status":"completed","details":f"{forecast_result['trend']} · {forecast_result['observations']} observations"})
        except HTTPException as exc:
            stages.append({"id":"forecast","label":"Forecast","status":"skipped","details":str(exc.detail)})
    else:
        stages.append({"id":"forecast","label":"Forecast","status":"skipped","details":"Date/value columns not selected"})

    explanation=explain(AutoMLRequest(rows=clean_rows,target=auto["target"],task=auto["task"]))
    stages.append({"id":"explain","label":"Explain","status":"completed","details":f"{len(explanation['global_drivers'])} global drivers analyzed"})

    decision_parts=[
        f"Autopilot analyzed {len(clean_rows)} cleaned rows and detected {auto['task']} with '{auto['target']}' as the target.",
        f"The model comparison selected {auto['best_model']} with a holdout score of {auto['best_score']}.",
    ]
    if anomaly_result:
        decision_parts.append(f"It flagged {anomaly_result['anomalies_found']} potentially unusual records.")
    if forecast_result:
        decision_parts.append(f"The recent time-series trend is {forecast_result['trend']}.")
    if auto["warnings"]:
        decision_parts.append(f"{len(auto['warnings'])} data-quality/leakage warning(s) should be reviewed before production use.")
    else:
        decision_parts.append("No high-risk leakage pattern was detected by the baseline checks.")
    report=" ".join(decision_parts)

    result={
        "version":"stage-7",
        "started_at":started,
        "completed_at":datetime.now(timezone.utc).isoformat(),
        "rows_original":len(df),
        "rows_cleaned":len(clean_rows),
        "columns":len(df.columns),
        "target":auto["target"],
        "task":auto["task"],
        "quality":clean_summary,
        "models":auto["models"],
        "best_model":auto["best_model"],
        "best_score":auto["best_score"],
        "feature_importance":auto["feature_importance"],
        "warnings":auto["warnings"],
        "anomalies":anomaly_result,
        "forecast":forecast_result,
        "explanation":explanation,
        "report":report,
        "stages":stages,
    }
    return result


class DriftRequest(BaseModel):
    baseline_rows:list[dict[str,Any]]
    current_rows:list[dict[str,Any]]
    threshold:Optional[float]=0.2
    baseline_version_id:Optional[int]=None

@app.post("/drift")
def drift(req:DriftRequest):
    base=frame(req.baseline_rows);cur=frame(req.current_rows)
    shared=[c for c in base.columns if c in cur.columns]
    if not shared: raise HTTPException(400,"Baseline and current datasets have no shared columns")
    results=[]
    for col in shared:
        a=base[col]; b=cur[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av=float(a.mean()) if len(a) else 0.0; bv=float(b.mean()) if len(b) else 0.0
            scale=float(a.std()) if pd.notna(a.std()) and float(a.std())>1e-9 else max(abs(av),1.0)
            magnitude=abs(bv-av)/scale
            metric="standardized_mean_shift"
        else:
            ap=a.fillna("__missing__").astype(str).value_counts(normalize=True)
            bp=b.fillna("__missing__").astype(str).value_counts(normalize=True)
            cats=set(ap.index)|set(bp.index)
            magnitude=0.5*sum(abs(float(ap.get(x,0))-float(bp.get(x,0))) for x in cats)
            metric="total_variation"
        results.append({"feature":str(col),"drift":round(float(magnitude),4),"metric":metric,"status":"alert" if magnitude>=req.threshold else "stable"})
    results.sort(key=lambda x:x["drift"],reverse=True)
    alerts=sum(x["status"]=="alert" for x in results)
    status="alert" if alerts else "stable"
    report=f"{alerts} of {len(results)} shared features exceeded the configured drift threshold."
    baseline_sig=dataset_signature(req.baseline_rows)
    current_sig=dataset_signature(req.current_rows)
    con=sqlite3.connect(DB_PATH)
    cur_db=con.execute("INSERT INTO monitoring_runs(created_at,baseline_version_id,baseline_signature,current_signature,baseline_rows,current_rows,threshold,status,alerts,report,features_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(),req.baseline_version_id,baseline_sig,current_sig,len(base),len(cur),req.threshold,status,alerts,report,json.dumps(results)))
    con.commit();run_id=cur_db.lastrowid;con.close()
    return {"id":run_id,"baseline_rows":len(base),"current_rows":len(cur),"threshold":req.threshold,"alerts":alerts,"features":results,"status":status,"report":report}
