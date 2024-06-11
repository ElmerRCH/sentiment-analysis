from pydantic import BaseModel
from model import SentimentModel
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/")
async def root(response: Response = Response()):
    response.status_code = 403
    return 'hola'

model = SentimentModel()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: TextRequest):
    prediction = model.predict(request.text)
    return {"sentiment": prediction}