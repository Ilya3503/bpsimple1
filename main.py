from fastapi import FastAPI
from app.camera import capture_pointcloud

app = FastAPI()


@app.get("/capture")
def capture():

    filepath = capture_pointcloud()

    return {
        "status": "ok",
        "file": filepath
    }