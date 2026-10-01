import {NextResponse} from "next/server";
export async function POST(req:Request){
  const base=process.env.ANALYSIS_API_URL;
  if(!base)return NextResponse.json({error:"ANALYSIS_API_URL is not configured. Use the browser profiler or connect the Python service."},{status:503});
  try{
    const upstream=await fetch(base.replace(/\/$/,"")+"/train",{method:"POST",headers:{"content-type":"application/json"},body:await req.text()});
    return new NextResponse(await upstream.text(),{status:upstream.status,headers:{"content-type":"application/json"}});
  }catch(e){return NextResponse.json({error:e instanceof Error?e.message:"Analysis service unavailable"},{status:502})}
}