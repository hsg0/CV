# FILE: lesson14.py
# LOCATION: python-CV/routes/opencv/lesson14.py
# WHY: Access Lesson 14 through our FastAPI backend.
# WHAT: Provides image processing and image retrieval endpoints.
# HOW: Connects URLs to Lesson 14 controller functions.
# WHEN: FastAPI receives a Lesson 14 request.
# WHERE: Imported and registered by main.py.

from fastapi import APIRouter

from controllers.openCV.lesson14 import (
    processFeatureMatching,
    getFeatureImage
)


# Create the Lesson 14 router.
lesson14Router = APIRouter()


# Process images using the selected head coordinates.
lesson14Router.get("/lesson14/process")(
    processFeatureMatching
)


# Retrieve a previously processed image.
lesson14Router.get("/lesson14/image")(
    getFeatureImage
)
