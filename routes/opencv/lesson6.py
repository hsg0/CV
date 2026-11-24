from fastapi import APIRouter
from controllers.openCV.lesson6 import drawARectangleAroundTheObject, getAnnotatedImage

lesson6Router = APIRouter()

lesson6Router.get("/lesson6/draw-rectangle")(drawARectangleAroundTheObject)
lesson6Router.get("/lesson6/annotated-image")(getAnnotatedImage)
