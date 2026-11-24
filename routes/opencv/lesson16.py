# FILE: lesson16.py
# LOCATION: python-CV/routes/opencv/lesson16.py
# WHY: Access Lesson 16 through our FastAPI backend.
# WHAT: Streams Webcam 0 with the selected processing mode.
# HOW: Connects the stream URL to the Lesson 16 controller.
# WHEN: The frontend opens the Lesson 16 stream.
# WHERE: Imported and registered by main.py.

from fastapi import APIRouter
from controllers.openCV.lesson16 import captureFrame, saveReference, streamVideo

lesson16Router = APIRouter()

lesson16Router.get("/lesson16/stream")(streamVideo)
lesson16Router.get("/lesson16/frame")(captureFrame)
lesson16Router.get("/lesson16/reference")(saveReference)
