from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv


from config import UPLOAD_FOLDER, RESULT_FOLDER, DEMO_FOLDER
from routes import router
load_dotenv()

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/results",    StaticFiles(directory=str(RESULT_FOLDER)), name="results")
app.mount("/demo_video", StaticFiles(directory=str(DEMO_FOLDER)),   name="demo_video")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
