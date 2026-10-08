# task-api

CRUD-сервис для управления задачами. FastAPI + Docker + автодеплой.

## Локальный запуск

    conda create -y -n task-api python=3.12
    conda activate task-api
    pip install -r requirements.txt
    uvicorn app.main:app --reload

## Живой сервис

Документация и swagger: https://185-46-10-97.nip.io/docs