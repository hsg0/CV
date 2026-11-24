from fastapi import APIRouter
from controllers.openCV.lesson1 import tryToGetCVVersion

lesson1Router = APIRouter()

lesson1Router.get("/lesson1/get-cv-version")(tryToGetCVVersion)
