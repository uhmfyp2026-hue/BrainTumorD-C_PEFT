from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def root():
    return {"status": "NeuroScan AI backend running"}

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    data = await file.read()
<<<<<<< HEAD
    return predict(data)
=======
    return predict(data)

>>>>>>> 1854273c38c7f10f3bd36904a32aa2a566058d52
