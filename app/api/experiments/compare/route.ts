import { NextResponse } from "next/server";

const API = process.env.ANALYSIS_API_URL || "http://localhost:8000";

export async function GET(req: Request) {
  const ids = new URL(req.url).searchParams.get("ids") || "";
  try {
    const response = await fetch(API + "/experiments/compare?ids=" + encodeURIComponent(ids), { cache: "no-store" });
    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch {
    return NextResponse.json({ error: "Experiment comparison service is unavailable." }, { status: 502 });
  }
}
