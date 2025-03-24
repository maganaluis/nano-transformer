# serve.py

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from nano_transformer.models import DummyTransformer

app = FastAPI(title="Dummy Transformer Serving API")

# Instantiate and set the model to evaluation mode
model = DummyTransformer(
    input_vocab_size=1000,
    output_vocab_size=1000,
)
model.eval()

class PredictionRequest(BaseModel):
    src: list[int]  # Source token sequence as a list of ints
    tgt: list[int]  # Target token sequence as a list of ints

@app.get("/")
def health_check():
    return {"message": "Dummy Transformer API is up and running."}

@app.post("/predict")
def predict(request: PredictionRequest):
    # Convert input lists to torch tensors with shape (seq_length, batch_size=1)
    src_tensor = torch.tensor(request.src).unsqueeze(1)
    tgt_tensor = torch.tensor(request.tgt).unsqueeze(1)
    
    with torch.no_grad():
        logits = model(src_tensor, tgt_tensor)
    
    # Remove the batch dimension and convert to a Python list
    logits_list = logits.squeeze(1).tolist()
    return {"logits": logits_list}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
