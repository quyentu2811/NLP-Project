import logging
import torch # dùng để sử dụng PyTorch
import torch.nn as nn
# sử dụng tokenizer tự động và mô hình chuỗi sang chuỗi, thích hợp cho tóm tắt văn bản
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# General class for BART and FLAN-T5
class GeneralModel:
    # Hàm khởi tạo để lưu checkpoint:
    def __init__(self, checkpoint, rank = 128):
        """
        Tạo một thiết bị dựa trên khả năng của hệ thống, và tải tokenizer và mô hình từ checkpoint đó
        """
        self.checkpoint = checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(self.device)
        self.rank = rank
        self.initialize_lora()

    def initialize_lora(self): 
        self.lora_apdaptations = nn.ModuleList([
            nn.Linear(self.base_model.config.hidden_size, self.rank, bias=False).to(self.device)
            for _ in range(self.base_model.config.num_hidden_layers)
        ])
        self.lora_revert = nn.ModuleList([
            nn.Linear(self.rank, self.base_model.config.hidden_size, bias=False).to(self.device)
            for _ in range(self.base_model.config.num_hidden_layers)
        ])
    def forward(self, input_ids):
        """
        Phương thức nhận văn bản đầu vào, mã hóa thành input_ids, sử dụng mô hình để sinh văn bản
        LoRa được áp dụng trong quá trình truyền tín hiệu qua mô hình
        """
        outputs = self.base_model(input_ids, output_hidden_states = True)
        new_hidden_states = []
        for i, hidden_state in enumerate(outputs.hidden_state):
            adapted_hidden_state = self.lora_apdaptations[i](hidden_state)
            adapted_hidden_state = self.lora_revert[i](adapted_hidden_state)
            new_hidden_states.append(hidden_state + adapted_hidden_state)
        
        outputs.last_hidden_state = torch.stack(new_hidden_states)
        return outputs.logits
    def generate(self, input_text, **kwargs):
        try:
            """
            Phương thức này nhận văn bản đầu vào và các tham số khác, sinh ra văn bản dựa trên mô 
            hình seq2seq. Văn bản đầu ra được sinh từ việc mã hóa văn bản đầu vào thành input_ids, 
            sử dụng mô hình để sinh ra các token mới, và giải mã các token này thành văn bản
            """
            logger.info(f"Generating output...")
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            outputs = self.forward(input_ids)
            generated_text = self.tokenizer.decode(outputs, skip_special_tokens = True)

            # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Summary: {generated_text}")

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