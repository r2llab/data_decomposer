from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from symphony.core.pipeline import Pipeline

app = FastAPI(title="Symphony Query Engine")
pipeline = Pipeline()

class Query(BaseModel):
    text: str

class Answer(BaseModel):
    result: str

@app.post("/query", response_model=Answer)
async def process_query(query: Query):
    """Process a natural language query and return the answer."""
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = pipeline.run_query(query.text)
        return Answer(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
