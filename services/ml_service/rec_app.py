"""FastAPI-приложение для рекомендации."""
from fastapi import FastAPI

from ml_service.rec_handler import FastApiHandler

"""
Пример запуска:
uvicorn ml_service.rec_app:app --reload --port 8081 --host 0.0.0.0

Для просмотра документации API и совершения тестовых запросов зайти на  http://127.0.0.1:8081/docs
"""

# создаём приложение FastAPI
app = FastAPI()
app.handler = FastApiHandler()

@app.get("/")
def read_root() -> dict:
    """
    Root endpoint that returns the status of the service.

    Returns:
        dict: A dictionary indicating the service status.
    """
    return {'Staus': 'Ok'}


@app.post('/get_rec') 
def get_rec(user_id: int, item_id: int):
    """Функция для получения предсказаний как от пользователя (item_id = -1) так и от пользователя в зависимости от того какой трек он слушает 
    Args:
        user_id (int): Идентификатор пользователя.
        item_id (int): Идентификатор трека.

    Returns:
        response (dict): Возвращает список рекомендации для пользователя.
    """
    all_params = {
        'user_id': user_id,
        'item_id': item_id
    }
    response = app.handler.handle(all_params)

    return response