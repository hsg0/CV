
from fastapi import FastAPI
import uvicorn
from routes.opencv.lesson1 import lesson1Router
from routes.opencv.lesson2 import lesson2Router
from routes.opencv.lesson3 import lesson3Router
from routes.opencv.lesson4 import lesson4Router
from routes.opencv.lesson5 import lesson5Router
from routes.opencv.lesson6 import lesson6Router
from routes.opencv.lesson7 import lesson7Router
from routes.opencv.lesson8 import lesson8Router
from routes.opencv.lesson9 import lesson9Router
from routes.opencv.lesson10 import lesson10Router
from routes.opencv.lesson11 import lesson11Router
from routes.opencv.lesson12 import lesson12Router
from routes.opencv.lesson13 import lesson13Router

app = FastAPI()


app.include_router(lesson1Router)
app.include_router(lesson2Router)
app.include_router(lesson3Router)
app.include_router(lesson4Router)
app.include_router(lesson5Router)
app.include_router(lesson6Router)
app.include_router(lesson7Router)
app.include_router(lesson8Router)
app.include_router(lesson9Router)
app.include_router(lesson10Router)
app.include_router(lesson11Router)
app.include_router(lesson12Router)
app.include_router(lesson13Router)


@app.get("/")
def home():
    return {"message": "Computer Vision API is running!"}


@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    print("Starting server on port 4020")
    print("http://127.0.0.1:4020")
    print("http://localhost:4020")
    uvicorn.run("main:app", host="127.0.0.1", port=4020, reload=True)