import logging
import torch # dùng để sử dụng PyTorch
# from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
# sử dụng tokenizer tự động và mô hình chuỗi sang chuỗi, thích hợp cho tóm tắt văn bản
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Config


logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


# General class for BART and FLAN-T5
class GeneralModel(nn.Module):
    # Hàm khởi tạo để lưu checkpoint:
    def __init__(self, checkpoint):
        """
        Tạo một thiết bị dựa trên khả năng của hệ thống, và tải tokenizer và mô hình từ checkpoint đó
        """
        super(GeneralModel, self).__init__()
        self.checkpoint = checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype = torch.bfloat16).to(self.device)
    
    def forward(self, input_text, **kwargs):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        outputs = self.base_model.generate(input_ids, **kwargs)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def generate(self, input_text, **kwargs):
        try:
            """
            Phương thức này nhận văn bản đầu vào và các tham số khác, sinh ra văn bản dựa trên mô 
            hình seq2seq. Văn bản đầu ra được sinh từ việc mã hóa văn bản đầu vào thành input_ids, 
            sử dụng mô hình để sinh ra các token mới, và giải mã các token này thành văn bản
            """
            generated_text = self.forward(input_text, **kwargs)
            logger.info(f"Summary: {generated_text}")
            return generated_text

            return generated_text

        except Exception as e:
            logger.error(f"Error while generating: {e}")
            raise e

# Các lớp chuyên biệt trên dùng để tải các phiên bản cụ thể của mô hình như Bart, Flan-t5 từ thư viện
# FLAN-T5 MODEL
class FlanT5Model(GeneralModel):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)


# BART MODEL
class BartModel(GeneralModel):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)  

def load_model(checkpoint):
    """
    Loads a model base on the `checkpoint` and optionally the `model_type`

    Args:
        checkpoint (str): the checkpoint from huggingface
        model_type (str, optional): Specific the model type (e.g. "bart" or "flan-t5")
    
    Returns:
        GeneralModel: The loaded model instance
    """
    try:
        if "bart" in checkpoint:
            logger.info(f"Load Bart model from checkpoint: {checkpoint}")
            return BartModel(checkpoint)
        
        if "flan" in checkpoint:
            logger.info(f"Load Flan-T5 model from checkpoint: {checkpoint}")
            return FlanT5Model(checkpoint)
        
        else:
            logger.info(f"Load general model from checkpoint: {checkpoint}")
            return GeneralModel(checkpoint)
        
    except Exception as e:
        logger.error("Error while loading model: {e}")
        raise e

# if __name__=='__main__':
#     checkpoint = "google/flan-t5-base"
#     model = load_model(checkpoint)
#     print(model)

#     prompt = "Summarize the following conversation:\n\n#Person1#: Tell me something about your\
#       Valentine's Day. #Person2#: Ok, on that day, boys usually give roses to the sweet hearts\
#         and girls give them chocolate in return. #Person1#: So romantic. young people must have\
#           lot of fun. #Person2#: Yeah, that is what the holiday is for, isn't it?\n\nSummary:"
    
#     output1 = model.generate(prompt, min_new_tokens=120, max_length=256)
#     output2 = model.generate(prompt, min_new_tokens=200, max_length=512)

#     print(output1)
#     print("\n\n")
#     print(output2)