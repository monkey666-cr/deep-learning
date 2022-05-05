from app import app


@app.task
def task01():
    print("task01")
