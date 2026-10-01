import { NextResponse } from "next/server";
const API = process.env.ANALYSIS_API_URL || "http://localhost:8000";
export async function POST(req: Request) {
  try {
    const response = await fetch(API + "/drift", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(await req.json()), cache:"no-store" });
    return NextResponse.json(await response.json(), {status:response.status});
  } catch { return NextResponse.json({error:"Drift monitoring service is unavailable."},{status:502}); }
}
