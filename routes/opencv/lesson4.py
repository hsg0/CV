from fastapi import APIRouter
from controllers.openCV.lesson4 import bringInImage

lesson4Router = APIRouter()

lesson4Router.get("/lesson4/bring-in-image")(bringInImage)
