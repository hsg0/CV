from fastapi import APIRouter
from controllers.openCV.lesson12 import getColorImage, processColorDetection

lesson12Router = APIRouter()

lesson12Router.get("/lesson12/detect-color")(processColorDetection)
lesson12Router.get("/lesson12/color-image")(getColorImage)
