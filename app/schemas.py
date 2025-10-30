from pydantic import BaseModel


class EchoRequest(BaseModel):
    message: str


class EchoResponse(BaseModel):
    echo: str


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
