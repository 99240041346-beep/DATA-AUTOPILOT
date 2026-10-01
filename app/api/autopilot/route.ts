import { NextResponse } from "next/server";

const API = process.env.ANALYSIS_API_URL || "http://localhost:8000";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const response = await fetch(API + "/autopilot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });
    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch {
    return NextResponse.json({ error: "Autopilot service is unavailable. Check the analysis API." }, { status: 502 });
  }
}
