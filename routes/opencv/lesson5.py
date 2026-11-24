from fastapi import APIRouter, HTTPException
from controllers.openCV.lesson5 import selectDogHead

lesson5Router = APIRouter()


def selectDogHeadRoute():
    try:
        savedFileName = selectDogHead()

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail="Image not found"
        ) from error

    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to save cropped image"
        ) from error

    if savedFileName is None:
        return {"message": "Window closed without saving"}

    return {
        "message": "Image saved",
        "file": savedFileName,
    }


lesson5Router.get("/lesson5/select-dog-head")(selectDogHeadRoute)
