from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent import investigate

app = FastAPI(
    title="Fraud Investigation Agent",
    description="Autonomous fraud investigation powered by LangGraph + Groq",
    version="1.0.0"
)


class InvestigateRequest(BaseModel):
    trans_num: str


class InvestigateResponse(BaseModel):
    trans_num: str
    summary: str
    report: dict | None


@app.post("/investigate", response_model=InvestigateResponse)
def run_investigation(request: InvestigateRequest):
    try:
        result = investigate(request.trans_num)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)