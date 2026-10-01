"use client";

import { useMemo, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, LineChart, Line
} from "recharts";

type Col = {
  name: string;
  type: "number" | "text";
  missing: number;
  unique: number;
  mean?: number;
  min?: number;
  max?: number;
};

type Dataset = {
  name: string;
  rows: number;
  cols: number;
  columns: Col[];
  preview: string[][];
  numeric: { name: string; mean: number; min: number; max: number }[];
  raw: Record<string, string | number | null>[];
};

const NAV_GROUPS = [
  { label: "WORKSPACE", items: [
    ["Mission", "◈", "Autonomous command center"],
    ["Overview", "⌂", "Dataset overview"],
    ["Data Profile", "▤", "Schema & quality"],
    ["EDA", "◫", "Explore the data"],
    ["AI insights", "✦", "Anomalies & forecasts"],
  ]},
  { label: "INTELLIGENCE", items: [
    ["ML Lab", "◎", "Train & compare models"],
    ["Explainability", "⌁", "Understand predictions"],
    ["Prediction", "◆", "Prediction studio"],
    ["What-if", "↯", "Decision simulation"],
  ]},
  { label: "OPERATIONS", items: [
    ["Autopilot", "⚡", "Run the full pipeline"],
    ["Monitoring", "◉", "Drift & data health"],
    ["Models", "◇", "Model registry"],
    ["History", "◴", "Experiments"],
    ["Reports", "⇩", "Export & reporting"],
  ]},
];

function parseCSV(text: string): string[][] {
  return text
    .replace(/\r/g, "")
    .split("\n")
    .filter(Boolean)
    .map((line) => {
      const cells: string[] = [];
      let quoted = false;
      let cell = "";
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"') quoted = !quoted;
        else if (ch === "," && !quoted) {
          cells.push(cell);
          cell = "";
        } else {
          cell += ch;
        }
      }
      cells.push(cell);
      return cells.map((v) => v.trim());
    });
}

function profile(name: string, grid: string[][]): Dataset {
  const headers = grid[0] || [];
  const rows = grid.slice(1);
  const columns: Col[] = headers.map((header, j) => {
    const values = rows.map((row) => row[j] ?? "");
    const numeric = values
      .filter((v) => v !== "" && Number.isFinite(Number(v)))
      .map(Number);
    const nonEmpty = values.filter(Boolean).length;
    const isNumber = numeric.length >= Math.max(3, nonEmpty * 0.7);
    return {
      name: header || `Column ${j + 1}`,
      type: isNumber ? "number" : "text",
      missing: values.filter((v) => !v).length,
      unique: new Set(values.filter(Boolean)).size,
      mean: isNumber ? numeric.reduce((a, b) => a + b, 0) / (numeric.length || 1) : undefined,
      min: isNumber && numeric.length ? Math.min(...numeric) : undefined,
      max: isNumber && numeric.length ? Math.max(...numeric) : undefined,
    };
  });
  const raw = rows.map((row) =>
    Object.fromEntries(
      headers.map((key, j) => {
        const value = row[j] ?? "";
        const n = Number(value);
        return [key, value !== "" && Number.isFinite(n) ? n : value || null];
      })
    )
  );
  return {
    name,
    rows: rows.length,
    cols: headers.length,
    columns,
    preview: [headers, ...rows.slice(0, 8)],
    numeric: columns
      .filter((c) => c.type === "number")
      .map((c) => ({
        name: c.name,
        mean: c.mean ?? 0,
        min: c.min ?? 0,
        max: c.max ?? 0,
      })),
    raw,
  };
}

async function postJSON(path: string, body: unknown) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return response.json();
}

function Stat({label, value}: {label: string; value: unknown}) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-950/50 p-4">
      <p className="text-xs uppercase tracking-widest text-slate-500">{label}</p>
      <p className="mt-2 truncate text-lg font-bold text-slate-100">{String(value ?? "—")}</p>
    </div>
  );
}

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`glass rounded-3xl p-6 ${className}`}>{children}</div>;
}

function ActionButton({
  children,
  onClick,
  disabled = false,
  secondary = false,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  secondary?: boolean;
}) {
  return (
    <button
      disabled={disabled}
      onClick={onClick}
      className={
        secondary
          ? "rounded-xl border border-slate-700 px-4 py-2 text-sm font-bold text-slate-300 disabled:opacity-40"
          : "rounded-xl bg-cyan-300 px-5 py-2.5 text-sm font-black text-slate-950 disabled:opacity-40"
      }
    >
      {children}
    </button>
  );
}

export default function Home() {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [tab, setTab] = useState("Overview");
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [target, setTarget] = useState("");
  const [autoResult, setAutoResult] = useState<any>(null);
  const [cleanResult, setCleanResult] = useState<any>(null);
  const [askResult, setAskResult] = useState<any>(null);
  const [explainResult, setExplainResult] = useState<any>(null);
  const [autopilotResult, setAutopilotResult] = useState<any>(null);
  const [whatIfResult, setWhatIfResult] = useState<any>(null);
  const [driftResult, setDriftResult] = useState<any>(null);
  const [models, setModels] = useState<any[]>([]);
  const [versions, setVersions] = useState<any[]>([]);
  const [baseline, setBaseline] = useState("");
  const [history, setHistory] = useState<any[]>([]);
  const [experiments, setExperiments] = useState<any[]>([]);
  const [selectedExperiments, setSelectedExperiments] = useState<number[]>([]);
  const [comparison, setComparison] = useState<any>(null);
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [mobileNav, setMobileNav] = useState(false);

  const chart = useMemo(
    () =>
      dataset?.numeric.slice(0, 8).map((x) => ({
        name: x.name.length > 12 ? x.name.slice(0, 12) + "…" : x.name,
        value: Number(x.mean.toFixed(2)),
      })) ?? [],
    [dataset]
  );

  const upload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setDataset(profile(file.name, parseCSV(String(reader.result))));
    reader.readAsText(file);
  };

  const run = async (fn: () => Promise<void>) => {
    setBusy(true); setStatusMessage("Running analysis…");
    try { await fn(); setStatusMessage("Analysis completed."); }
    catch (error) { setStatusMessage(error instanceof Error ? error.message : "Analysis could not be completed."); }
    finally { setBusy(false); }
  };

  const runAutoML = () =>
    run(async () => {
      const result = await postJSON("/api/automl", {
        rows: dataset?.raw,
        target: target || null,
        task: "auto",
      });
      setAutoResult(result);
      if (!result.error) {
        try {
          await postJSON("/api/models", {
            name: `${dataset?.name} · AutoPilot candidate`,
            rows: dataset?.raw,
            result,
          });
        } catch {}
      }
    });

  const runAsk = () =>
    run(async () => {
      setAskResult(
        await postJSON("/api/ask", {
          rows: dataset?.raw,
          question: prompt,
        })
      );
    });

  const runClean = () =>
    run(async () => {
      setCleanResult(
        await postJSON("/api/clean", {
          rows: dataset?.raw,
          target: target || null,
        })
      );
    });

  const runAnomalies = () =>
    run(async () => {
      setCleanResult(await postJSON("/api/anomalies", { rows: dataset?.raw }));
    });

  const runExplain = () =>
    run(async () => {
      setExplainResult(
        await postJSON("/api/explain", {
          rows: dataset?.raw,
          target: target || null,
          task: "auto",
        })
      );
    });

  const runAutopilot = () =>
    run(async () => {
      const date = (document.getElementById("auto-date") as HTMLSelectElement | null)?.value || null;
      const value = (document.getElementById("auto-value") as HTMLSelectElement | null)?.value || null;
      setAutopilotResult(
        await postJSON("/api/autopilot", {
          rows: dataset?.raw,
          target: target || null,
          task: "auto",
          date_column: date,
          value_column: value,
          forecast_periods: 7,
          run_anomalies: true,
          run_forecast: true,
        })
      );
    });

  const runPrediction = () =>
    run(async () => {
      if (!dataset || !target) return;
      const changes: Record<string, string> = {};
      dataset.columns.filter((c) => c.name !== target && c.type === "number").forEach((col) => {
        const element = document.getElementById(`pred-${col.name}`) as HTMLInputElement | null;
        if (element && element.value !== "") changes[col.name] = element.value;
      });
      setPredictionResult(await postJSON("/api/what-if", { rows: dataset.raw, target, task: "auto", changes }));
    });

  const runWhatIf = () =>
    run(async () => {
      if (!dataset) return;
      const changes: Record<string, string> = {};
      dataset.columns.filter((c) => c.name !== target).slice(0, 10).forEach((col) => {
        const element = document.getElementById(`wf-${col.name}`) as HTMLInputElement | null;
        if (element) changes[col.name] = element.value;
      });
      setWhatIfResult(
        await postJSON("/api/what-if", {
          rows: dataset.raw,
          target: target || null,
          task: "auto",
          changes,
        })
      );
    });

  const refreshModels = () =>
    run(async () => {
      const result = await fetch("/api/models").then((r) => r.json());
      setModels(result.models || []);
    });

  const saveVersion = () =>
    run(async () => {
      if (!dataset) return;
      const result = await postJSON("/api/datasets/versions", {
        name: `${dataset.name} · ${new Date().toLocaleString()}`,
        rows: dataset.raw,
      });
      const list = await fetch("/api/datasets/versions").then((r) => r.json());
      setVersions(list.versions || []);
      if (result.id) setBaseline(String(result.id));
    });

  const compareVersion = () =>
    run(async () => {
      if (!dataset || !baseline) return;
      setDriftResult(
        await postJSON("/api/drift/version", {
          baseline_version_id: Number(baseline),
          current_rows: dataset.raw,
          threshold: 0.2,
        })
      );
    });

  const loadMonitoring = () =>
    run(async () => {
      const result = await fetch("/api/monitoring/history").then((r) => r.json());
      setHistory(result.runs || []);
    });

  const loadExperiments = () =>
    run(async () => {
      const result = await fetch("/api/experiments").then((r) => r.json());
      setExperiments(result.experiments || []);
    });

  const compareExperiments = () =>
    run(async () => {
      const result = await fetch(
        "/api/experiments/compare?ids=" + selectedExperiments.join(",")
      ).then((r) => r.json());
      setComparison(result);
    });

  if (!dataset) {
    return (
      <main className="min-h-screen grid-bg">
        <header className="border-b border-slate-800/80 bg-[#07111f]/90 backdrop-blur-xl">
          <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
            <Brand />
            <label className="cursor-pointer rounded-xl bg-cyan-300 px-4 py-2 text-sm font-bold text-slate-950">
              <input type="file" accept=".csv,.txt" className="hidden" onChange={upload} />
              Upload dataset
            </label>
          </div>
        </header>
        <section className="mx-auto max-w-7xl px-6 py-16">
          <div className="grid gap-6 lg:grid-cols-[1.3fr_.7fr]">
            <Card>
              <p className="text-sm font-semibold text-cyan-300">FUTURE DATA WORKSPACE</p>
              <h2 className="mt-3 text-4xl font-black tracking-tight md:text-6xl">
                From raw data to <span className="text-cyan-300">decisions.</span>
              </h2>
              <p className="mt-5 max-w-2xl text-slate-400">
                DATA AUTOPILOT profiles data, cleans quality issues, runs machine-learning experiments,
                detects anomalies, explains predictions, monitors drift and turns analysis into decisions.
              </p>
              <div className="mt-7 flex flex-wrap gap-2">
                {["Profiling", "EDA", "AutoML", "Explainability", "What-if", "Monitoring"].map((x) => (
                  <span key={x} className="rounded-full border border-slate-700 bg-slate-900/60 px-3 py-1 text-xs text-slate-300">
                    {x}
                  </span>
                ))}
              </div>
            </Card>
            <Card>
              <p className="text-xs uppercase tracking-widest text-slate-500">Autopilot status</p>
              <div className="mt-5 flex items-center gap-3">
                <span className="h-3 w-3 rounded-full bg-emerald-400" />
                <b>Ready</b>
              </div>
              <p className="mt-3 text-sm leading-6 text-slate-400">
                Browser profiling starts instantly. Connect the FastAPI analysis service for ML workloads.
              </p>
              <label className="mt-7 inline-flex cursor-pointer rounded-2xl bg-slate-100 px-6 py-3 font-bold text-slate-950">
                <input type="file" accept=".csv,.txt" className="hidden" onChange={upload} />
                Choose CSV
              </label>
            </Card>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="min-h-screen grid-bg">
      <div className="mx-auto flex max-w-[1600px]">
        <aside className={`fixed inset-y-0 left-0 z-40 w-72 border-r border-slate-800 bg-[#06101d]/98 p-5 backdrop-blur-xl transition-transform lg:sticky lg:top-0 lg:h-screen lg:translate-x-0 ${mobileNav ? "translate-x-0" : "-translate-x-full"}`}>
          <div className="flex h-full flex-col">
            <div className="flex items-center justify-between">
              <Brand />
              <button className="rounded-lg border border-slate-800 px-2 py-1 text-slate-400 lg:hidden" onClick={() => setMobileNav(false)}>×</button>
            </div>
            <div className="mt-8 rounded-2xl border border-cyan-400/10 bg-cyan-400/5 p-4">
              <p className="text-[10px] font-bold uppercase tracking-[.2em] text-cyan-300">ACTIVE DATASET</p>
              <p className="mt-2 truncate text-sm font-bold text-slate-200">{dataset.name}</p>
              <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                <div className="rounded-xl bg-slate-950/60 p-2"><span className="text-slate-500">Rows</span><br/><b>{dataset.rows.toLocaleString()}</b></div>
                <div className="rounded-xl bg-slate-950/60 p-2"><span className="text-slate-500">Cols</span><br/><b>{dataset.cols}</b></div>
              </div>
            </div>
            <nav className="mt-7 flex-1 space-y-6 overflow-y-auto pr-1">
              {NAV_GROUPS.map((group) => (
                <div key={group.label}>
                  <p className="mb-2 px-3 text-[10px] font-bold tracking-[.2em] text-slate-600">{group.label}</p>
                  <div className="space-y-1">
                    {group.items.map(([name, icon, hint]) => (
                      <button key={name} title={hint} onClick={() => { setTab(name); setMobileNav(false); }}
                        className={`group flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm transition ${tab === name ? "bg-cyan-300 font-bold text-slate-950 shadow-lg shadow-cyan-300/10" : "text-slate-400 hover:bg-slate-900 hover:text-slate-100"}`}>
                        <span className="grid h-7 w-7 place-items-center rounded-lg bg-slate-950/40 text-sm">{icon}</span>
                        <span className="min-w-0 flex-1"><span className="block">{name}</span><span className={`block truncate text-[10px] ${tab === name ? "text-slate-700" : "text-slate-600"}`}>{hint}</span></span>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </nav>
            <div className="border-t border-slate-800 pt-4">
              <div className="flex gap-2">
                <label className="flex-1 cursor-pointer rounded-xl bg-cyan-300 px-3 py-2.5 text-center text-xs font-black text-slate-950">
                  <input type="file" accept=".csv,.txt" className="hidden" onChange={upload} />Replace dataset
                </label>
                <button onClick={() => setDataset(null)} className="rounded-xl border border-slate-800 px-3 text-xs text-slate-400">Reset</button>
              </div>
              <p className="mt-3 text-center text-[10px] text-slate-600">DATA AUTOPILOT · Autonomous Data Science</p>
            </div>
          </div>
        </aside>
        {mobileNav && <button aria-label="Close navigation" className="fixed inset-0 z-30 bg-black/60 lg:hidden" onClick={() => setMobileNav(false)} />}
        <section className="min-w-0 flex-1 px-4 py-5 sm:px-6 lg:px-8">
          <div className="mb-5 flex items-center justify-between gap-3 lg:hidden">
            <button onClick={() => setMobileNav(true)} className="rounded-xl border border-slate-800 bg-slate-950/70 px-3 py-2 text-sm text-slate-300">☰ Menu</button>
            <span className="truncate text-xs text-slate-500">{dataset.name}</span>
          </div>
          <div className="mb-6 flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/70 pb-5">
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[.2em] text-cyan-300">DATA AUTOPILOT / {tab}</p>
              <h2 className="mt-1 text-2xl font-black">{tab === "Mission" ? "Autonomous Command Center" : tab}</h2>
            </div>
            <div className="flex items-center gap-2">
              <span className={`h-2 w-2 rounded-full ${busy ? "bg-amber-300 animate-pulse" : "bg-emerald-400"}`} />
              <span className="text-xs text-slate-500">{busy ? "Processing" : "System ready"}</span>
            </div>
          </div>

        <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            ["Rows", dataset.rows],
            ["Columns", dataset.cols],
            ["Numeric", dataset.numeric.length],
            ["Missing cells", dataset.columns.reduce((sum, c) => sum + c.missing, 0)],
          ].map(([label, value]) => (
            <Card key={String(label)} className="p-5">
              <p className="text-xs uppercase tracking-widest text-slate-500">{label}</p>
              <p className="mt-2 text-3xl font-black">{String(value)}</p>
            </Card>
          ))}
        </div>

        {tab === "Mission" && (
          <div className="space-y-6">
            <Card>
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div><p className="text-xs font-bold uppercase tracking-widest text-cyan-300">AUTONOMOUS MISSION</p>
                <h2 className="mt-1 text-3xl font-black">One dataset. One mission.</h2>
                <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">Automatically profile, clean, train, detect anomalies, forecast, explain and summarize the dataset.</p></div>
                <ActionButton disabled={busy} onClick={runAutopilot}>{busy ? "Running mission…" : "Run Full Autopilot"}</ActionButton>
              </div>
              {statusMessage && <p className="mt-4 rounded-xl border border-slate-800 bg-slate-950/50 p-3 text-sm text-slate-400">{statusMessage}</p>}
            </Card>
            <div className="grid gap-6 lg:grid-cols-3">
              <Card><p className="text-xs uppercase tracking-widest text-slate-500">Dataset quality</p><p className="mt-2 text-4xl font-black">{Math.max(0,100-Math.round((dataset.columns.reduce((s,c)=>s+c.missing,0)/Math.max(1,dataset.rows*dataset.cols))*100))}/100</p><p className="mt-2 text-sm text-slate-500">Baseline missing-cell quality score.</p></Card>
              <Card><p className="text-xs uppercase tracking-widest text-slate-500">Target</p><select value={target} onChange={e=>setTarget(e.target.value)} className="mt-3 w-full rounded-xl border border-slate-800 bg-slate-950 p-3 text-sm"><option value="">Auto-detect</option>{dataset.columns.map(c=><option key={c.name} value={c.name}>{c.name}</option>)}</select></Card>
              <Card><p className="text-xs uppercase tracking-widest text-slate-500">Detected signals</p><div className="mt-3 space-y-2 text-sm text-slate-300"><div>Numeric: <b>{dataset.numeric.length}</b></div><div>Date-like: <b>{dataset.columns.filter(c=>/date|time|year|month|day/i.test(c.name)).length}</b></div><div>Missing: <b>{dataset.columns.reduce((s,c)=>s+c.missing,0)}</b></div></div></Card>
            </div>
            {autopilotResult && <Card><h3 className="text-xl font-bold">Mission result</h3><div className="mt-5 grid gap-3 md:grid-cols-3"><Stat label="Task" value={autopilotResult.task}/><Stat label="Target" value={autopilotResult.target}/><Stat label="Best model" value={autopilotResult.best_model}/></div><div className="mt-5 rounded-2xl border border-cyan-900/60 bg-cyan-950/20 p-5"><p className="text-xs uppercase tracking-widest text-cyan-300">Decision summary</p><p className="mt-2 leading-7 text-slate-200">{autopilotResult.report}</p></div><div className="mt-5 grid gap-3 md:grid-cols-2 lg:grid-cols-3">{(autopilotResult.stages||[]).map((stage:any)=><div key={stage.id} className="rounded-2xl border border-slate-800 bg-slate-950/50 p-4"><div className="flex justify-between"><b>{stage.label}</b><span className={stage.status==="completed"?"text-emerald-300":"text-amber-300"}>{stage.status}</span></div><p className="mt-2 text-xs text-slate-500">{stage.details}</p></div>)}</div></Card>}
          </div>
        )}

        {tab === "Overview" && (
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <h3 className="font-bold">Numeric signals</h3>
              <div className="mt-5 h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chart}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: "#0b1728", border: "1px solid #334155" }} />
                    <Bar dataKey="value" fill="#67e8f9" radius={[7, 7, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
            <Card>
              <h3 className="font-bold">Ask Autopilot</h3>
              <p className="mt-1 text-sm text-slate-500">Ask a direct question about the current dataset.</p>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Find the strongest drivers..."
                className="mt-5 h-28 w-full rounded-2xl border border-slate-800 bg-slate-950/50 p-4 text-sm outline-none"
              />
              <ActionButton disabled={busy || !prompt.trim()} onClick={runAsk}>
                {busy ? "Analyzing…" : "Ask Autopilot"}
              </ActionButton>
              {askResult && (
                <div className="mt-5 rounded-2xl bg-slate-950/50 p-4 text-sm">
                  {askResult.error ? (
                    <p className="text-amber-300">{askResult.error}</p>
                  ) : (
                    <>
                      <p className="leading-6 text-slate-300">{askResult.answer}</p>
                      <div className="mt-3 grid gap-2 sm:grid-cols-2">
                        {(askResult.insights || []).slice(0, 8).map((item: any, i: number) => (
                          <div key={i} className="rounded-xl bg-slate-900 p-3 text-xs text-slate-400">
                            {Object.entries(item).map(([key, value]) => (
                              <span key={key} className="mr-3">
                                <b className="text-slate-300">{key}:</b> {String(value)}
                              </span>
                            ))}
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}
            </Card>
          </div>
        )}

        {tab === "Data Profile" && (
          <div className="space-y-6">
            <Card className="overflow-hidden">
              <h3 className="mb-5 text-xl font-bold">Dataset profile</h3>
              <div className="overflow-auto">
                <table className="w-full text-left text-sm">
                  <thead className="border-b border-slate-800 text-xs uppercase text-slate-500">
                    <tr>
                      <th className="p-4">Column</th>
                      <th>Type</th><th>Missing</th><th>Unique</th><th>Mean</th><th>Range</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dataset.columns.map((c) => (
                      <tr key={c.name} className="border-b border-slate-900">
                        <td className="p-4 font-semibold">{c.name}</td>
                        <td>{c.type}</td>
                        <td>{c.missing}</td>
                        <td>{c.unique}</td>
                        <td>{c.mean === undefined ? "—" : c.mean.toFixed(3)}</td>
                        <td>
                          {c.min === undefined ? "—" : `${c.min.toFixed(2)} → ${c.max?.toFixed(2)}`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
            <Card>
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <p className="text-xs uppercase tracking-widest text-cyan-300">Stage 2</p>
                  <h3 className="mt-1 text-2xl font-bold">Autonomous Data Cleaner</h3>
                </div>
                <ActionButton disabled={busy} onClick={runClean}>
                  {busy ? "Scanning…" : "Analyze & Clean"}
                </ActionButton>
              </div>
              {cleanResult && <ResultBox result={cleanResult} />}
            </Card>
          </div>
        )}

        {tab === "EDA" && (
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <h3 className="font-bold">Dataset preview</h3>
              <div className="mt-5 overflow-auto">
                <table className="min-w-full text-left text-xs">
                  <tbody>
                    {dataset.preview.map((row, i) => (
                      <tr key={i} className="border-b border-slate-900">
                        {row.map((value, j) => <td key={j} className="whitespace-nowrap px-3 py-2">{value || "—"}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
            <Card>
              <h3 className="font-bold">Numeric baseline</h3>
              <div className="mt-5 h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chart}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: "#0b1728", border: "1px solid #334155" }} />
                    <Line type="monotone" dataKey="value" stroke="#67e8f9" strokeWidth={3} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>
        )}

        {tab === "AI insights" && (
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <p className="text-xs uppercase tracking-widest text-cyan-300">Stage 3</p>
              <h3 className="mt-2 text-2xl font-bold">Anomaly detection</h3>
              <p className="mt-2 text-sm text-slate-400">Scan numeric features with the backend Isolation Forest pipeline.</p>
              <ActionButton disabled={busy} onClick={runAnomalies}>
                {busy ? "Scanning…" : "Find anomalies"}
              </ActionButton>
              {cleanResult?.anomalies_found !== undefined && (
                <p className="mt-5 text-slate-300">
                  <span className="text-3xl font-black text-cyan-300">{cleanResult.anomalies_found}</span>{" "}
                  anomalies found ({Math.round((cleanResult.anomaly_rate || 0) * 100)}%).
                </p>
              )}
            </Card>
            <Card>
              <p className="text-xs uppercase tracking-widest text-cyan-300">Forecasting</p>
              <h3 className="mt-2 text-2xl font-bold">Trend forecast</h3>
              <div className="mt-5 grid gap-2">
                <select id="forecast-date" className="rounded-xl border border-slate-700 bg-slate-900 px-3 py-2 text-sm">
                  {dataset.columns.map((c) => <option key={c.name}>{c.name}</option>)}
                </select>
                <select id="forecast-value" className="rounded-xl border border-slate-700 bg-slate-900 px-3 py-2 text-sm">
                  {dataset.columns.map((c) => <option key={c.name}>{c.name}</option>)}
                </select>
                <ActionButton
                  disabled={busy}
                  onClick={() =>
                    run(async () => {
                      const date = (document.getElementById("forecast-date") as HTMLSelectElement).value;
                      const value = (document.getElementById("forecast-value") as HTMLSelectElement).value;
                      setCleanResult(await postJSON("/api/forecast", {
                        rows: dataset.raw, date_column: date, value_column: value, periods: 7
                      }));
                    })
                  }
                >
                  {busy ? "Forecasting…" : "Generate forecast"}
                </ActionButton>
              </div>
              {cleanResult?.trend && (
                <div className="mt-5 rounded-2xl bg-slate-950/50 p-4">
                  <b className="text-cyan-300">{cleanResult.trend}</b> trend · {cleanResult.observations} observations
                </div>
              )}
            </Card>
          </div>
        )}

        {tab === "ML Lab" && (
          <div className="space-y-6">
            <Card>
              <p className="text-xs uppercase tracking-widest text-cyan-300">AutoML experiment engine</p>
              <h3 className="mt-2 text-3xl font-black">Train, compare and select</h3>
              <p className="mt-2 max-w-3xl text-slate-400">
                Automatically detect the task and target, compare candidate models, inspect metrics and surface feature importance.
              </p>
              <div className="mt-6 flex flex-wrap gap-3">
                <select value={target} onChange={(e) => setTarget(e.target.value)} className="min-w-56 rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-sm">
                  <option value="">Auto-detect target</option>
                  {dataset.columns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
                </select>
                <ActionButton disabled={busy} onClick={runAutoML}>
                  {busy ? "Running AutoPilot…" : "Run AutoPilot"}
                </ActionButton>
              </div>
            </Card>
            {autoResult && <AutoMLResult result={autoResult} />}
          </div>
        )}

        {tab === "Explainability" && (
          <Card>
            <p className="text-xs uppercase tracking-widest text-cyan-300">Stage 4 · Explainable AI</p>
            <h3 className="mt-2 text-3xl font-black">Why did the model make this prediction?</h3>
            <p className="mt-2 max-w-3xl text-slate-400">Inspect global drivers and local effects for the latest record.</p>
            <div className="mt-5">
              <ActionButton disabled={busy} onClick={runExplain}>
                {busy ? "Explaining…" : "Explain prediction"}
              </ActionButton>
            </div>
            {explainResult && <ResultBox result={explainResult} />}
          </Card>
        )}

        {tab === "Prediction" && (
          <Card><p className="text-xs uppercase tracking-widest text-cyan-300">PREDICTION STUDIO</p><h3 className="mt-1 text-2xl font-black">Generate a prediction</h3><p className="mt-2 text-sm text-slate-500">Provide feature values and compare the scenario against the current baseline record.</p>
          <div className="mt-5 grid gap-4 md:grid-cols-2">{dataset.columns.filter(c=>c.name!==target&&c.type==="number").slice(0,12).map(c=><label key={c.name} className="text-sm text-slate-400">{c.name}<input id={`pred-${c.name}`} defaultValue={c.mean??""} type="number" className="mt-2 w-full rounded-xl border border-slate-800 bg-slate-950 p-3 text-slate-200"/></label>)}</div>
          <ActionButton disabled={busy||!target} onClick={runPrediction}>{busy?"Predicting…":"Generate prediction"}</ActionButton>
          {predictionResult&&<div className="mt-6 grid gap-4 md:grid-cols-3"><Stat label="Baseline" value={predictionResult.baseline}/><Stat label="Scenario" value={predictionResult.scenario}/><Stat label="Impact" value={predictionResult.impact}/></div>}
          {!target&&<p className="mt-3 text-xs text-amber-300">Choose a target in Mission first.</p>}</Card>
        )}

        {tab === "What-if" && (
          <Card>
            <p className="text-xs uppercase tracking-widest text-cyan-300">Stage 5 · Model-backed simulation</p>
            <h3 className="mt-2 text-3xl font-black">Test a decision before making it</h3>
            <div className="mt-6 grid gap-3 md:grid-cols-2">
              {dataset.columns.filter((c) => c.name !== target).slice(0, 10).map((col) => (
                <label key={col.name} className="text-sm text-slate-400">
                  {col.name}
                  <input
                    id={`wf-${col.name}`}
                    defaultValue={dataset.raw[dataset.raw.length - 1]?.[col.name] ?? ""}
                    className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"
                  />
                </label>
              ))}
            </div>
            <div className="mt-5">
              <ActionButton disabled={busy} onClick={runWhatIf}>{busy ? "Simulating…" : "Run scenario"}</ActionButton>
            </div>
            {whatIfResult && <ResultBox result={whatIfResult} />}
          </Card>
        )}

        {tab === "Autopilot" && (
          <div className="space-y-6">
            <Card>
              <p className="text-xs uppercase tracking-widest text-cyan-300">Stage 7 · Autonomous Data Scientist</p>
              <h3 className="mt-2 text-3xl font-black">Run the whole analysis automatically</h3>
              <p className="mt-2 max-w-3xl text-slate-400">
                Profile, clean, train, detect anomalies, forecast when possible, explain and produce a decision report.
              </p>
              <div className="mt-6 grid gap-3 md:grid-cols-3">
                <label className="text-sm text-slate-400">
                  Target
                  <select value={target} onChange={(e) => setTarget(e.target.value)} className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3">
                    <option value="">Auto-detect</option>
                    {dataset.columns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
                  </select>
                </label>
                <label className="text-sm text-slate-400">
                  Date column
                  <select id="auto-date" className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3">
                    <option value="">None</option>
                    {dataset.columns.map((c) => <option key={c.name}>{c.name}</option>)}
                  </select>
                </label>
                <label className="text-sm text-slate-400">
                  Forecast value
                  <select id="auto-value" className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3">
                    <option value="">None</option>
                    {dataset.columns.map((c) => <option key={c.name}>{c.name}</option>)}
                  </select>
                </label>
              </div>
              <div className="mt-5">
                <ActionButton disabled={busy} onClick={runAutopilot}>
                  {busy ? "Running full pipeline…" : "Run Full Autopilot"}
                </ActionButton>
              </div>
            </Card>
            {autopilotResult && <AutoMLResult result={autopilotResult} />}
          </div>
        )}

        {tab === "Monitoring" && (
          <div className="space-y-6">
            <Card>
              <p className="text-xs uppercase tracking-widest text-cyan-300">Stage 9 · Production monitoring</p>
              <h3 className="mt-2 text-3xl font-black">Dataset versioning & drift</h3>
              <div className="mt-5 flex flex-wrap gap-3">
                <ActionButton disabled={busy} onClick={saveVersion}>{busy ? "Saving…" : "Save current version"}</ActionButton>
                <ActionButton secondary onClick={loadMonitoring}>Load monitoring history</ActionButton>
              </div>
              <div className="mt-5 grid gap-4 md:grid-cols-[1fr_auto]">
                <select value={baseline} onChange={(e) => setBaseline(e.target.value)} className="rounded-xl border border-slate-700 bg-slate-950 px-4 py-3">
                  <option value="">Select baseline version</option>
                  {versions.map((v) => <option key={v.id} value={String(v.id)}>#{v.id} · {v.name} · {v.rows_used} rows</option>)}
                </select>
                <ActionButton disabled={!baseline || busy} onClick={compareVersion}>Compare current data</ActionButton>
              </div>
              {driftResult && <ResultBox result={driftResult} />}
            </Card>
            {history.length > 0 && (
              <Card>
                <h3 className="font-bold">Monitoring history</h3>
                <div className="mt-4 space-y-2">
                  {history.slice(0, 10).map((x) => (
                    <div key={x.id} className="flex flex-wrap justify-between gap-3 rounded-xl bg-slate-950/50 p-3 text-sm">
                      <span>Run #{x.id} · {new Date(x.created_at).toLocaleString()}</span>
                      <span className={x.status === "alert" ? "text-amber-300" : "text-emerald-300"}>{x.status}</span>
                    </div>
                  ))}
                </div>
              </Card>
            )}
          </div>
        )}

        {tab === "Models" && (
          <Card>
            <p className="text-xs uppercase tracking-widest text-cyan-300">Stage 12 · Model lifecycle</p>
            <h3 className="mt-2 text-3xl font-black">Model Registry</h3>
            <div className="mt-5">
              <ActionButton disabled={busy} onClick={refreshModels}>Refresh models</ActionButton>
            </div>
            <div className="mt-6 space-y-3">
              {models.length === 0 ? (
                <p className="rounded-2xl bg-slate-950/50 p-5 text-sm text-slate-500">No registered model versions yet. Run AutoML first.</p>
              ) : models.map((m) => (
                <div key={m.id} className="rounded-2xl border border-slate-800 bg-slate-950/50 p-5">
                  <div className="flex flex-wrap justify-between gap-3">
                    <div>
                      <p className="font-bold">{m.name}</p>
                      <p className="text-xs text-slate-500">v{m.id} · {m.rows_used} rows · {m.target || "—"}</p>
                    </div>
                    <span className="text-cyan-300">{m.status}</span>
                  </div>
                  <div className="mt-4 grid gap-3 sm:grid-cols-3">
                    <div><p className="text-xs text-slate-500">Model</p><p>{m.model_name}</p></div>
                    <div><p className="text-xs text-slate-500">Score</p><p className="text-cyan-300">{m.score ?? "—"}</p></div>
                    <ActionButton
                      secondary
                      disabled={m.status === "production" || busy}
                      onClick={() => run(async () => {
                        await fetch(`/api/models/${m.id}/promote`, { method: "POST" });
                        await refreshModels();
                      })}
                    >
                      {m.status === "production" ? "Production" : "Promote"}
                    </ActionButton>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )}

        {tab === "Reports" && (
          <Card>
            <p className="text-xs uppercase tracking-widest text-cyan-300">Stage 11 · Reporting</p>
            <h3 className="mt-2 text-3xl font-black">Portable analysis report</h3>
            <p className="mt-2 text-slate-400">Export the current analysis state as JSON or the dataset as CSV.</p>
            <div className="mt-6 flex flex-wrap gap-3">
              <ActionButton onClick={() => {
                const payload = {
                  generated_at: new Date().toISOString(),
                  dataset: dataset.name,
                  rows: dataset.raw.length,
                  columns: dataset.columns.map((x) => x.name),
                  autopilot: autopilotResult,
                  question: askResult,
                  monitoring: driftResult,
                };
                const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }));
                const a = document.createElement("a");
                a.href = url; a.download = "data-autopilot-report.json"; a.click();
                URL.revokeObjectURL(url);
              }}>Download JSON report</ActionButton>
              <ActionButton secondary onClick={() => {
                const headers = dataset.columns.map((x) => x.name);
                const escape = (v: unknown) => JSON.stringify(v ?? "");
                const csv = [headers.map(escape).join(","), ...dataset.raw.map((row) => headers.map((h) => escape(row[h])).join(","))].join("\n");
                const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
                const a = document.createElement("a");
                a.href = url; a.download = "data-autopilot-export.csv"; a.click();
                URL.revokeObjectURL(url);
              }}>Export CSV</ActionButton>
            </div>
          </Card>
        )}

        {tab === "History" && (
          <Card>
            <div className="flex flex-wrap justify-between gap-4">
              <div>
                <p className="text-xs uppercase tracking-widest text-cyan-300">Stage 6 · Experiment memory</p>
                <h3 className="mt-2 text-3xl font-black">Experiment Registry</h3>
              </div>
              <ActionButton disabled={busy} onClick={loadExperiments}>Refresh history</ActionButton>
            </div>
            <div className="mt-5 flex flex-wrap gap-3">
              <ActionButton disabled={selectedExperiments.length < 2 || busy} onClick={compareExperiments}>
                Compare selected ({selectedExperiments.length})
              </ActionButton>
              {comparison && <span className="self-center text-sm text-slate-400">Showing {comparison.count} experiments</span>}
            </div>
            <div className="mt-6 space-y-3">
              {experiments.length === 0 ? (
                <p className="rounded-2xl bg-slate-950/50 p-5 text-sm text-slate-500">No saved experiments yet.</p>
              ) : experiments.map((x) => (
                <label key={x.id} className="block cursor-pointer rounded-2xl border border-slate-800 bg-slate-950/50 p-5">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <span className="flex items-center gap-2 text-xs text-slate-500">
                      <input
                        type="checkbox"
                        checked={selectedExperiments.includes(x.id)}
                        onChange={(e) => setSelectedExperiments((current) =>
                          e.target.checked ? [...current, x.id] : current.filter((id) => id !== x.id)
                        )}
                      />
                      compare
                    </span>
                    <b>{x.name}</b>
                    <span className="text-cyan-300">{x.best_model || "analysis"}</span>
                  </div>
                </label>
              ))}
            </div>
          </Card>
        )}
        </section>
      </div>
    </main>
  );
}

function Brand() {
  return (
    <div className="flex items-center gap-3">
      <div className="grid h-9 w-9 place-items-center rounded-xl bg-cyan-400/15 font-black text-cyan-300 ring-1 ring-cyan-300/30">DA</div>
      <div>
        <h1 className="font-bold">DATA AUTOPILOT</h1>
        <p className="text-[11px] uppercase tracking-[.25em] text-slate-500">Autonomous Data Science</p>
      </div>
    </div>
  );
}

function ResultBox({ result }: { result: any }) {
  if (!result) return null;
  if (result.error) return <div className="mt-6 rounded-2xl border border-amber-300/20 bg-amber-300/5 p-4 text-sm text-amber-300">{result.error}</div>;
  return (
    <div className="mt-6 grid gap-4 md:grid-cols-2">
      {[
        ["Status", result.status],
        ["Target", result.target],
        ["Task", result.task],
        ["Model", result.best_model],
        ["Score", result.best_score],
        ["Report", result.report],
      ].filter(([, value]) => value !== undefined && value !== null).map(([label, value]) => (
        <div key={label} className="rounded-2xl bg-slate-950/50 p-4">
          <p className="text-xs uppercase tracking-widest text-slate-500">{label}</p>
          <p className="mt-2 text-sm leading-6 text-slate-300">{String(value)}</p>
        </div>
      ))}
      {result.features && (
        <div className="rounded-2xl bg-slate-950/50 p-4 md:col-span-2">
          <p className="text-xs uppercase tracking-widest text-slate-500">Signals</p>
          <div className="mt-3 space-y-2">
            {result.features.slice(0, 10).map((x: any, i: number) => (
              <div key={x.feature || i} className="flex justify-between text-sm">
                <span>{x.feature}</span><span className="text-cyan-300">{x.drift ?? x.importance ?? x.status}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function AutoMLResult({ result }: { result: any }) {
  if (result.error) return <ResultBox result={result} />;
  return (
    <div className="grid gap-6 lg:grid-cols-[1.1fr_.9fr]">
      <Card>
        <p className="text-xs uppercase tracking-widest text-slate-500">Selected model</p>
        <p className="mt-2 text-2xl font-black text-cyan-300">{result.best_model || "—"}</p>
        <p className="mt-1 text-sm text-slate-400">{result.task || "analysis"} · target: {result.target || "—"}</p>
        <div className="mt-5 space-y-2">
          {(result.models || []).map((model: any) => (
            <div key={model.model} className="flex items-center justify-between rounded-xl bg-slate-950/50 p-4">
              <div>
                <p className="font-semibold">{model.model}</p>
                <p className="text-xs text-slate-500">
                  {Object.entries(model.metrics || {}).map(([k, v]) => `${k}: ${v}`).join(" · ")}
                </p>
              </div>
              <span className="font-bold text-cyan-300">{model.score}</span>
            </div>
          ))}
        </div>
      </Card>
      <Card>
        <p className="text-xs uppercase tracking-widest text-cyan-300">Decision report</p>
        <p className="mt-4 leading-7 text-slate-300">{result.report || "Analysis completed."}</p>
        <h4 className="mt-6 font-bold">Top signals</h4>
        <div className="mt-3 space-y-2">
          {(result.feature_importance || []).slice(0, 8).map((x: any) => (
            <div key={x.feature} className="flex justify-between rounded-xl bg-slate-950/50 px-3 py-2 text-sm">
              <span>{x.feature}</span><span className="text-cyan-300">{x.importance}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}