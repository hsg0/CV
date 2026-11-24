from fastapi import APIRouter
from controllers.openCV.lesson3 import webcamImageToGrayScale

lesson3Router = APIRouter()

lesson3Router.get("/lesson3/webcam-image-to-grayscale")(webcamImageToGrayScale)
