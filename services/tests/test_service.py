import os
from io import BytesIO

import requests
import random

import boto3
import pandas

from dotenv import load_dotenv
load_dotenv()


def load_df_from_s3(key: str):
    '''Скачивание из s3 и перевод в pandas.DataFrame. Формат в котором должен храниться файл - parquet

    Args:
        key: ключ в s3
    
    Returns:
        df: данные в формате pandas.DataFrame
    '''
    S3_BUCKET_NAME = 's3-student-mle-20240522-d840b46cf7'
    ENDPOINT_URL = 'https://storage.yandexcloud.net'

    s3_resource = boto3.resource('s3',
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )

    parquet_buffer = BytesIO()
    s3_object = s3_resource.Object(S3_BUCKET_NAME, key)
    s3_object.download_fileobj(parquet_buffer)
    df = pandas.read_parquet(parquet_buffer)
    return df

#пользователей будем брать случайным образом из rank_features/user.parquet
users_list = list(load_df_from_s3('recsys/recommendations/rank_features/user.parquet')['user_id'])

#треки возьмем такие у которых есть достаточно похожие аналоги (score похожести > 0.1)
items_list = load_df_from_s3('recsys/recommendations/similar.parquet')
items_list = items_list[items_list['score'].apply(lambda x: x[1] > 0.1 )]
items_list = list(items_list['item_id'])

def make_one_rec(new_user_prob=0.1, with_no_item_prob=0.1):
    """Функция случайным образом берет строку из "datasets/flats_dataset.csv" и отправляет запрос на сервер
    
    Args:
        error_prob (float): вероятность отправки пустых параметров (имитация ошибки)

    Returns:
        response (response): Ответ от сервера
    """
    if random.random() < new_user_prob:
        user_id = -1
    else:
        user_id = random.choice(users_list)

    if random.random() < with_no_item_prob:
        item_id = -1
    else:
        item_id = random.choice(items_list)
    
    response = requests.post(f'http://84.252.142.60:8081/get_rec?user_id={user_id}&item_id={item_id}')
    return response


   

with open('tests/test_service.log', 'w') as f:
    for i in range(500):
        f.write(make_one_rec().content.decode())
        f.write('\n')
        