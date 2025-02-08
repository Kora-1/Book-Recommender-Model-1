from fastapi import FastAPI
import model

app = FastAPI()

# Ensure the model is loaded when the app starts
@app.on_event("startup")
async def load_model():
    model.load()
    print("Model Loaded Successfully")

@app.get("/")
async def root():
    return {"message": "API Made By Dishant"}

# @app.get("/run")
# async def say_hello():
#     # Optionally, you could call `load()` here if you want to re-load the model on request
#     # But it's better to call it once at startup, as done in the `startup` event
#     return {"message": "Model Loaded Successfully (via /run endpoint)"}

# @app.get("/recommend_books/{query}/{length}")
# async def get_books(query: str, length: int = 5):  # Use default length 5
#     recommendation = model.recommend_books(query, length)
#     return recommendation

@app.get("/recommend_books/{query}")
async def get_books(query: str, length: int | None = 5):
    recommendation = model.recommend_books(query, length)
    return recommendation

# Uncomment if you're running uvicorn directly from this script
# uvicorn.run("main:app", host="0.0.0.0", port=8000)
