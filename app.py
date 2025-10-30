from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class EchoReq(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/echo")
def echo(req: EchoReq):
    return {"echo": req.message}