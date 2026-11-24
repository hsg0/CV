from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from controllers.openCV.lesson10 import erodeMask

lesson10Router = APIRouter()


def erodeMaskRoute():
    try:
        savePath = erodeMask()

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail="Image not found"
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to save image"
        ) from error

    return FileResponse(
        path=savePath,
        media_type="image/png"
    )


lesson10Router.get("/lesson10/erode-mask")(erodeMaskRoute)
