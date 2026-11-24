# FILE: lesson18.py
# LOCATION: python-CV/routes/opencv/lesson18.py
# WHY: Access Lesson 18 through our FastAPI backend.
# WHAT: Provides live video streaming, live mode switching, camera stop, and reference selection.
# HOW: Connects Lesson 18 HTTP endpoints to controller functions.
# WHEN: The frontend webpage interacts with Lesson 18.
# WHERE: Imported and registered by main.py.

from fastapi import APIRouter

from controllers.openCV.lesson18 import (
    captureReferenceFrame,
    saveReferenceSelection,
    stopVideoStream,
    streamVideo,
    updateVideoSettings,
)

lesson18Router = APIRouter()

lesson18Router.get("/lesson18/stream")(streamVideo)
lesson18Router.get("/lesson18/settings")(updateVideoSettings)
lesson18Router.get("/lesson18/stop")(stopVideoStream)
lesson18Router.get("/lesson18/frame")(captureReferenceFrame)
lesson18Router.get("/lesson18/reference")(saveReferenceSelection)
