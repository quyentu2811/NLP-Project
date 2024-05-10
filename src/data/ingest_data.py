# Đoạn code trên dùng để thiết kế để tải và trả về một tập dữ liệu từ một nguồn dữ liệu cụ thể
import logging
from datasets import load_dataset # là hàm dùng để tải tập dữ liệu từ kho dữ liệu
from datasets import DatasetDict # cấu trúc dữ liệu để lưu tập dữ liệu

# logging.basicConfig(level=logging.INFO) # ghi nhật kí cấp độ thông tin
# logger = logging.getLogger(__name__)
# lớp bao bọc quá trình tải dữ liệu
class IngestDataset:
    # hàm lưu đường dẫn hoặc tên của tập dữ liệu
    def __init__(self, datapath: str="knkarthick/dialogsum") -> None:
        self.datapath = datapath

    def get_data(self) -> DatasetDict:
        logger.info(f"Loading data from {self.datapath}")
        # load dataset với datapath đã cho
        data = load_dataset(self.datapath, trust_remote_code=True) # cho phép thực thi mã từ xa
        logger.info(f"Complete loading data from {self.datapath}")
        
        return data
    
# tạo thực thể cảu lớp IngestDataset và gọi phương thức get_data để tải tập dữ liệu
def ingest_data(datapath: str) -> DatasetDict:
    try:
        ingest_data = IngestDataset(datapath)
        dataset = ingest_data.get_data()

        return dataset
    
    except Exception as e:
        logger.error(f"Error while loading data: {e}")
        raise e
    

# if __name__=='__main__':
#     datapath = "knkarthick/dialogsum"
#     dataset = ingest_data(datapath)
#     print(dataset)
#     print(type(dataset))