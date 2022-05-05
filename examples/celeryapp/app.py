from celery import Celery

app = Celery()
app.config_from_object("settings")
app.autodiscover_tasks(["task00", "task01"])
