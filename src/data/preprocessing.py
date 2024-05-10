# Các câu lệnh trên thể hiện quy trình tiền xử lý dữ liệu
import logging

import os # tương tác với hệ điều hành
import sys # tương tác với biến hệ thống

from datasets import DatasetDict # lưu trữ tập dữ liệu

# Dùng để tải và sử dụng bộ tách từ (tokenizer) đã được đào tạo sẵn
from transformers import AutoTokenizer 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# lấy đường dẫn tuyệt đối của thư mục chứa tệp hiện tại
path = os.path.abspath(os.path.dirname(__file__))
# Thêm đường dẫn này vào đầu của danh sách đường dẫn mà Python sử dụng để tìm kiếm các module khi nhập
sys.path.insert(0, path)

from data_strategy import *
from ingest_data import *

# checkpoint = "google/flan-t5-base"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

class DataPreprocessing:
    def __init__(self, data: DatasetDict, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy

    def handle_data(self, *args):
        try:
            return self.strategy.handle_data(self.data, *args)
        
        except Exception as e:
            logger.error(f"Error while preprocessing data: {e}")
            raise e
        
# Hàm để tiền xử lý tập dữ liệu sử dụng một chiến lược tách từ cụ thể
def preprocessing_data(data: DatasetDict, tokenizer, *args) -> DatasetDict:
    try:
        # Tạo mội đối tượng với bộ tách từ tokenizer đã cho
        tokenizing_strategy = DataTokenizingStrategy(tokenizer)
        # tạo một đối tượng với tập dữ liệu và chiến lược tách từ
        data_preprocess = DataPreprocessing(data, tokenizing_strategy)
        # lấy dữ liệu đã được xử lý từ đối tượng DataPreprocessing và trả về nó
        tokenized_data = data_preprocess.handle_data()

        return tokenized_data

    except Exception as e:
        logger.error(f"Error while pre-processing data: {e}")
        raise e
    
# if __name__=='__main__':
#     checkpoint = "google/flan-t5-base"
#     datapath = "knkarthick/dialogsum"

#     tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
#     data = ingest_data(datapath)

#     data = preprocessing_data(data, tokenizer)

#     print(data)