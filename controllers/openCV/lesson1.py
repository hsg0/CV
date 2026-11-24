# ../controllers/openCV/lesson1.py

from config.open_cv.configureCV import configureCV
from fastapi import HTTPException

openCV = configureCV()


async def tryToGetCVVersion():
    try:
        return {
            "openCVVersion": openCV.getVersionString()
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve OpenCV version"
        ) from error

