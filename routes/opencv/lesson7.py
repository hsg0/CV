from fastapi import APIRouter
from controllers.openCV.lesson7 import applyGaussianBlurToAnImage

lesson7Router = APIRouter()

lesson7Router.get("/lesson7/apply-blur")(applyGaussianBlurToAnImage)
