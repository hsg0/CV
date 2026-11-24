from fastapi import APIRouter
from controllers.openCV.lesson13 import getTransformedImage, processImageTransformations

lesson13Router = APIRouter()

lesson13Router.get("/lesson13/transform-image")(processImageTransformations)
lesson13Router.get("/lesson13/transformed-image")(getTransformedImage)
