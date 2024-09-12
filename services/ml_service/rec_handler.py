"""Класс FastApiHandler, который обрабатывает запросы API."""
import os
from io import BytesIO
from dotenv import load_dotenv

import boto3
import pickle
import pandas

load_dotenv()

S3_BUCKET_NAME = 's3-student-mle-20240522-d840b46cf7'
ENDPOINT_URL = 'https://storage.yandexcloud.net'

s3_resource = boto3.resource('s3',
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
)

class FastApiHandler:
    """Класс FastApiHandler, который обрабатывает запрос и возвращает предсказание."""

    def __init__(self):
        """Инициализация переменных класса."""
        
        # модель ранжирующая рекомендации
        self.rank_model = self.load_model_from_s3('recsys/recommendations/rank_model.pkl')

        # признаки для модели ранжирования
        self.rank_feature_user = self.load_df_from_s3('recsys/recommendations/rank_features/user.parquet').set_index('user_id')
        self.rank_feature_item = self.load_df_from_s3('recsys/recommendations/rank_features/item.parquet').set_index('item_id')
        self.rank_feature_user_item = self.load_df_from_s3('recsys/recommendations/rank_features/user_item.parquet').set_index(['user_id', 'item_id'])
        
        # различные рекомендации
        self.pop_rec = self.load_df_from_s3('recsys/recommendations/top_popular.parquet').head(2000)
        self.als_rec = self.load_df_from_s3('recsys/recommendations/personal_als.parquet').set_index('user_id')
        self.sim_items = self.load_df_from_s3('recsys/recommendations/similar.parquet').set_index('item_id')

    def load_df_from_s3(self, key: str):
        '''Скачивание из s3 и перевод в pandas.DataFrame. Формат в котором должен храниться файл - parquet

        Args:
            key: ключ в s3
        
        Returns:
            df: данные в формате pandas.DataFrame
        '''
        try:
            parquet_buffer = BytesIO()
            s3_object = s3_resource.Object(S3_BUCKET_NAME, key)
            s3_object.download_fileobj(parquet_buffer)
            df = pandas.read_parquet(parquet_buffer)
            return df
        
        except Exception as e:
            print(f'Failed to load DataFrame: {e}')

    def load_model_from_s3(self, key: str):
        '''Функция для скачивания модели

        Args:
            key: ключ в s3
        
        Returns:
            model (model.pkl)
        '''
        try:
            buffer = BytesIO()
            s3_object = s3_resource.Object(S3_BUCKET_NAME, key)
            s3_object.download_fileobj(buffer)
            buffer.seek(0)
            return pickle.load(buffer)

        except Exception as e:
            print(f'Failed to load model: {e}')

    def get_pop_rec(self, top_n: int = 10):
        '''Получение рекомендации по принципу самые популярные треки

        Args:
            top_n (int): количество лучших треков
        
        Returns:
            rec_list: список рекомендации
        '''
        pop_rec = self.pop_rec.head(top_n)
        return list(pop_rec['item_id'])

    def get_als_rec(self, user_id: int):
        '''Получение персональных рекомендации ALS

        Args:
            user_id (int): идентификатор пользователя
        
        Returns:
            rec_list: список рекомендации
        '''
        if user_id in self.als_rec.index:
            rec_row = self.als_rec.loc[user_id]
            return list(rec_row['item_id'])
        else:
            # при отсутствии данных о пользователе выдает пустой список
            return []

    def get_sim_items(self, item_id: int):
        '''Получение рекомендации по принципу похожести треков друг на друга (i2i)

        Args:
            item_id (int): идентификатор трека
        
        Returns:
            rec_list: список рекомендации
        '''
        if item_id in self.sim_items.index:
            sim_row = self.sim_items.loc[item_id]
            return list(sim_row['item_id_sim'])[1:]
        else:
            # при отсутствии данных о треке выдает пустой список
            return []
    
    def get_all_rec(self, user_id: int, item_id: int):
        '''Получение рекомендации из всех источников (топ популярные, персональные, i2i)

        Args:
            user_id (int): идентификатор пользователя
            item_id (int): идентификатор трека
        
        Returns:
            rec_list: список рекомендации
        '''
        all_rec = dict()
        all_rec['pop_rec'] = self.get_pop_rec()
        all_rec['als_rec'] = self.get_als_rec(user_id)
        all_rec['sim_items'] = self.get_sim_items(item_id)
        return all_rec
        
    def get_rank_params(self, user_id: int, item_id: int):
        '''Для каждой пары (юзер, трек) получаем признаки для модели ранжирования, используя которые модель даст значение для ранжирования

        Args:
            user_id (int): идентификатор пользователя
            item_id (int): идентификатор трека
        
        Returns:
            params (dict): список рекомендации
        '''        
        # изначально в params значения, которыми при обучении модели заполняли пропуски
        params = {
            'als_score': 0,
            'name_len': 0,
            'main_genre': 0,
            'top_num': 99999,
            'count': 0
        }
        if user_id in self.rank_feature_user.index:
            feature_user = self.rank_feature_user.loc[user_id]
            params['main_genre'] = feature_user.loc['main_genre']
            params['count'] = feature_user.loc['count']
        if item_id in self.rank_feature_item.index:
            feature_item = self.rank_feature_item.loc[item_id]
            params['top_num'] = feature_item.loc['top_num']
            params['name_len'] = feature_item.loc['name_len']
        if (user_id, item_id) in self.rank_feature_user_item.index:
            feature_user_item = self.rank_feature_user_item.loc[user_id, item_id]
            params['als_score'] = feature_user_item.loc['als_score']
        return params


    def rank_rec(self, user_id: int, rec_list: list):
        '''Для пользователя и списка возможных рекомендации получаем список значений для ранжирования

        Args:
            user_id (int): идентификатор пользователя
            rec_list (list): списка возможных рекомендации
        
        Returns:
            rank_list (list): список значений для ранжирования
        '''       
        rank_list = []
        for item in rec_list:
            params = self.get_rank_params(user_id, item)
            rank = self.rank_model.predict_proba(list(params.values()))[1]
            rank_list.append(rank)
        return rank_list

    def get_best_rec(self, user_id: int, item_id: int, best_n: int = 10):
        '''Для пользователя и трека (который сейчас слушает пользователь) мы делаем рекомендации и ранжируем их. Далее берем Далее берем лучшие согласно ранжированию

        Args:
            user_id (int): идентификатор пользователя
            item_id (int): идентификатор трека
            best_n (int): количетво выдоваемых рекомендации после ранжирования
        
        Returns:
            rec (dict): список рекомендации и источники этих рекомендации
        '''    
        all_rec_dict = self.get_all_rec(user_id, item_id)
        all_rec = []
        for rec_list in all_rec_dict.values():
            all_rec += rec_list

        rank = self.rank_rec(user_id, all_rec)

        rank_rec = list(zip(all_rec, rank))
        rank_rec.sort(key=lambda x: x[1])
        rec_list = [int(rec[0]) for rec in rank_rec[:best_n]]

        #источник рекомендации rec_from
        rec_from = dict()
        for key in all_rec_dict:
            rec_from[key] = 0
        for item in rec_list:
            for key in all_rec_dict:
                if item in all_rec_dict[key]:
                    rec_from[key] += 1
        
        return {'rec_from': rec_from, 'rec_list': rec_list}

    def handle(self, params):
        """Функция для обработки запросов API.
        
        Args:
            params (dict): Словарь параметров запроса.
        
        Returns:
            dict: Словарь, содержащий результат выполнения запроса.
        """
        try:
            # Валидируем запрос к API
            user_id = params['user_id']
            item_id = params['item_id']
            
            rec = self.get_best_rec(user_id, item_id)
            response = {
                'user_id': user_id, 
                'item_id': item_id,
                'rec_from': rec['rec_from'],
                'rec_list': rec['rec_list']
            }
        except Exception as e:
            print(f'Error while handling request: {e}')
            return {'Error': 'Problem with request'}
        else:
            print(response)
            return response
        






if __name__ == '__main__':
    a = FastApiHandler()
    print()
    a.handle({'user_id': 1374582, 'item_id': 60064065})
    a.handle({'user_id': 1374582, 'item_id': -1})
    a.handle({'user_id': -1, 'item_id': 60064065})
    print()
    
