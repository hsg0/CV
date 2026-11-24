from fastapi import APIRouter
from controllers.openCV.lesson8 import applyCannyEdgeDetection

lesson8Router = APIRouter()

lesson8Router.get("/lesson8/canny-edges")(applyCannyEdgeDetection)
