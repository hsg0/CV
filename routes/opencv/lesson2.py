from fastapi import APIRouter
from controllers.openCV.lesson2 import getCameraFrame

lesson2Router = APIRouter()

lesson2Router.get("/lesson2/get-camera-frame")(getCameraFrame)
