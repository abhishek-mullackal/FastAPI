from pydantic import BaseModel


class SentimentData(BaseModel):
    text: str
