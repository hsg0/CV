from fastapi import APIRouter
from controllers.openCV.lesson9 import getProcessedImage, processImage

lesson9Router = APIRouter()

lesson9Router.get("/lesson9/process-image")(processImage)
lesson9Router.get("/lesson9/processed-image")(getProcessedImage)
