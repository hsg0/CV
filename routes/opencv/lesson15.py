# FILE: lesson15.py
# LOCATION: python-CV/routes/opencv/lesson15.py
# WHY: Access Lesson 15 through our FastAPI backend.
# WHAT: Provides the live webcam stream.
# HOW: Connects the stream URL to the Lesson 15 controller.
# WHEN: FastAPI receives a Lesson 15 request.
# WHERE: Imported and registered by main.py.

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from controllers.openCV.lesson15 import streamVideo


# Create the Lesson 15 router.
lesson15Router = APIRouter()


# Show the live camera in a normal browser page.
def watchCamera():
    return HTMLResponse(
        """
        <img src="/lesson15/stream" alt="Camera">
        """
    )


lesson15Router.get("/lesson15")(watchCamera)


# Stream webcam frames to the browser.
lesson15Router.get("/lesson15/stream")(
    streamVideo
)
