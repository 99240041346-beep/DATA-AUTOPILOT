import {NextResponse} from "next/server";

async function proxy(request:Request, method:"GET"|"POST"){
  const base=process.env.ANALYSIS_API_URL;
  if(!base) return NextResponse.json({error:"Analysis service is not configured."},{status:503});
  try{
    const body=method==="POST"?JSON.stringify(await request.json()):undefined;
    const upstream=await fetch(base.replace(/\/$/,"")+"/models",{method,headers:{"Content-Type":"application/json"},body,cache:"no-store"});
    const data=await upstream.json().catch(()=>({error:"Invalid analysis service response"}));
    return NextResponse.json(data,{status:upstream.status});
  }catch{return NextResponse.json({error:"Unable to reach the analysis service."},{status:502});}
}
export async function GET(request:Request){return proxy(request,"GET");}
export async function POST(request:Request){return proxy(request,"POST");}
