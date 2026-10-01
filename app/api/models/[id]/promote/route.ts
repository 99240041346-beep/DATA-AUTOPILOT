import {NextResponse} from "next/server";
export async function POST(request:Request,{params}:{params:Promise<{id:string}>}){
  const base=process.env.ANALYSIS_API_URL;
  if(!base) return NextResponse.json({error:"Analysis service is not configured."},{status:503});
  try{
    const {id}=await params;
    const upstream=await fetch(base.replace(/\/$/,"")+"/models/"+encodeURIComponent(id)+"/promote",{method:"POST",cache:"no-store"});
    const data=await upstream.json().catch(()=>({error:"Invalid analysis service response"}));
    return NextResponse.json(data,{status:upstream.status});
  }catch{return NextResponse.json({error:"Unable to reach the analysis service."},{status:502});}
}
