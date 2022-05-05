app_name = "examples"

broker_url = "redis://localhost:6379/0"

result_backend = "redis://localhost:6379/1"

task_serializer = "json"

result_serializer = "json"

accept_content = ["json"]

task_default_queue = f"{app_name}.queue"

task_routes = {
    "task00.tasks.task00": {
        "queue": "task00"
    },
    "task00.tasks.task01": {
        "queue": "task01"
    }
}
