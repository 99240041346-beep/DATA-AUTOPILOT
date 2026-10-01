import {NextResponse} from "next/server";

export async function POST(request:Request){
  const base=process.env.ANALYSIS_API_URL;
  if(!base) return NextResponse.json({error:"Analysis service is not configured."},{status:503});
  try{
    const body=await request.json();
    const upstream=await fetch(base.replace(/\/$/,"")+"/clean",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body),cache:"no-store"});
    return NextResponse.json(await upstream.json(),{status:upstream.status});
  }catch{return NextResponse.json({error:"Unable to reach the analysis service."},{status:502});}
}