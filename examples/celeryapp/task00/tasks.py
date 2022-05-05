from app import app


@app.task
def task00():
    print("task00")
