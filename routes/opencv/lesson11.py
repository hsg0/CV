from fastapi import APIRouter
from controllers.openCV.lesson11 import getContoursImage, processContours

lesson11Router = APIRouter()

lesson11Router.get("/lesson11/process-contours")(processContours)
lesson11Router.get("/lesson11/contours-image")(getContoursImage)
