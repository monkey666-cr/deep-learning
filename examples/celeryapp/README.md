启动celery

```shell
celery -A celeryapp.app worker --concurrency=1 --loglevel=INFO -Q task00,task01
```