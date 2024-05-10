import logging
from abc import ABC, abstractclassmethod

from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
from ingest_data import ingest_data
# Khởi tạo để ghi lại các thông báo trên màn hình từ mức độ Info trở lên
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

"""
DataStrategy là một lớp trừu tượng trong Python, được sử dụng như một khuôn mẫu cơ bản cho các lớp khác để kế thừa và triển khai. 
Lớp này sử dụng module abc (Abstract Base Classes) để định nghĩa các phương thức trừu tượng mà các lớp con bắt buộc phải triển khai. 
Điều này giúp đảm bảo rằng tất cả các lớp con của DataStrategy sẽ có cùng một giao diện cho việc xử lý dữ liệu, 
nhưng có thể có cách thực hiện khác nhau tùy thuộc vào nhu cầu cụ thể.
"""
class DataStrategy(ABC):
    """
    Abstract class for handling data
    """
    @abstractclassmethod
    def handle_data(self, data, *args) -> None:
        pass


class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: Dataset, *args) -> DatasetDict: # kế thừa từ lớp DataStrategy
        """
        If loaded dataset is not divided, this method is used to split data using 
        Dataset.train_test_split() of hugging face (But it is not implement cause all the volume in dataset is divided)
        Cách thức triển khai: sử dụng các phương pháp có sẵn ví dụ như train_test_split từ
        thư viện datasets để chia dữ liệu
        """
        try:
            pass

        except Exception as e:
            logger.error(f"Error while dividing data: {e}")
            raise e
        

class DataTokenizingStrategy(DataStrategy):
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
    # Khởi tạo với một bộ tách từ và định nghĩa phương thức handle_data để tách dữ liệu

    def handle_data(self, data: DatasetDict, *args) -> DatasetDict:
        try:
            # thiết lập kết thúc câu (eos_token) làm token đệm (pad_token): dùng cho BART
            self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Tokenizing dataset!")
            """
            Dùng phương thức map để áp dụng hàm preprocess_function trên mỗi mẫu dữ liệu
            trong tập dữ liệu, được thực hiện theo batched (batched: xử lý dữ liệu theo từng
            lô để tăng hiệu quả xử lý)
            """
            tokenized_dataset = data.map(self.preprocess_function, batched=True)

            logger.info(f"Removing unnecessary columns!")
            # Loại bỏ các cột không cần thiết (cột thông tin phụ: bình luật)
            tokenized_dataset = tokenized_dataset.remove_columns([key for key in data["train"][0].keys()])

            # tokenized_dataset = tokenized_dataset.filter(lambda example, index: index%100==0, with_indices=True)

            return tokenized_dataset

        except Exception as e:
            logger.info(f"Error while tokenizing data: {e}")
            raise e
        
    def preprocess_function(self, data: Dataset, *args) -> Dataset:
        prefix = "Summarize the following conversation:\\nn"
        suffix = "\n\nSummary: "
        # Tạo ra các chuỗi đầu vào bằng cách thêm hai tiền tố vào mỗi đoạn hội thoại
        inputs = [prefix + input + suffix for input in data["dialogue"]]

        # Chuyển đổi các chuỗi đầu vào thành token
        data["input_ids"] = self.tokenizer(inputs, padding="max_length", truncation=True, return_tensors="pt").input_ids
        """
        input_ids là một mảng của các số nguyên mà mỗi số đại diện cho một từ hoặc một phần của từ (token)
        trong một chuỗi đầu vào. Mỗi token được ánh xạ đến một ID duy nhất trong bộ từ vựng mà mô hình đã
        được huấn luyện trên đó. Quá trình này biến đổi văn bản thô thành dạng mà mô hình có thể xử lý được.
        Ví dụ: Giả sử chuỗi đầu vào là "Hello, world!". Nếu bộ tách từ (tokenizer) đã được đào tạo để nhận
        biết từ "Hello" và dấu phẩy và từ "world" cùng với dấu chấm than, quá trình tách từ có thể cho kết
        quả là ['Hello', ',', 'world', '!']. Mỗi token này sẽ được chuyển thành một ID, ví dụ input_ids có
        thể là [7592, 16, 592, 84].
        """
        data["attention_mask"] = self.tokenizer(inputs, padding="max_length", truncation=True, return_tensors="pt").attention_mask
        """
        attention_mask là một mảng mà mỗi phần tử của nó chỉ ra liệu một token cụ thể có nên được mô hình
        chú ý đến không khi tính toán các phép toán của mô hình. Thông thường, các token hợp lệ được đánh
        dấu là 1 và các token đệm (padding) được đánh dấu là 0. Mục đích của mặt nạ chú ý là để mô hình
        không tính toán sự ảnh hưởng từ các token đệm, vì chúng không mang thông tin có ích.
        Ví dụ: Nếu input_ids là [7592, 16, 592, 84, 0, 0, 0] (với 0 là token đệm để đồng bộ hóa độ dài của
        các chuỗi), thì attention_mask tương ứng sẽ là [1, 1, 1, 1, 0, 0, 0].
        """
        data["labels"] = self.tokenizer(data["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
        """
        labels dùng trong quá trình huấn luyện mô hình để cung cấp đích đến hoặc kết quả mong muốn mà mô 
        hình cần học để dự đoán
        -> labels là thông tin mục tiêu mà mô hình cần học để dự đoán chính xác hơn
        labels cần được chuyển đổi thành input_ids. Nhưng trong quá trình huấn luyện, các token đệm trong 
        nhãn không nên ảnh hưởng đến việc tính doán mất mát của mô hình. 
        """
        label_ignore_ids = []
        for label in data["labels"]:
            label_example = [l if l != 0 else -100 for l in label]
            label_ignore_ids.append(label_example)
        # Câu lệnh trên để xử lí việc này: token đệm được gán với giá trị -100 để mô hình biết là bỏ qua 
        # chúng khi tính toán mất mát
        data["labels"] = label_ignore_ids
        """
        Nếu nhãn là "This is a summary.", và sau khi chuyển đổi, labels là [20920, 250, 21, 10845, 84, 0, 0, 0],
        thì để mô hình không xem xét các token đệm khi tính mất mát, nhãn được điều chỉnh thành
        [20920, 250, 21, 10845, 84, -100, -100, -100].
        """

        return data

# if __name__=='__main__':
#     checkpoint = "google/flan-t5-base"
#     datapath = "knkarthick/dialogsum"

#     tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#     dataset = ingest_data(datapath)
#     print(dataset["train"])

#     print("\n\n\n")
#     tokenizing_data = DataTokenizingStrategy(tokenizer)
#     tokenized_dataset = tokenizing_data.handle_data(dataset)

#     print(tokenized_dataset)

#     print(type(tokenized_dataset))

#     print(type(tokenized_dataset["train"]))

#     print(tokenized_dataset["train"][0])