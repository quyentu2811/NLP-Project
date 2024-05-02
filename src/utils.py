import argparse

import os
import sys
import yaml

import logging

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback

# path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, path)

# from src.evaluate.rouge_metric import compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# định nghĩa và phân tích các tham số dòng lệnh để cấu hình quá trình huấn luyện.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine tuning LLM for Dialogue Text Summarization")
    parser.add_argument("--configpath", type=str, default=None)
    parser.add_argument("--huggingface_hub_token", type=str, default=None)
    parser.add_argument("--wandb_token", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="google/flan-t5-base")
    parser.add_argument("--datapath", type=str, default="knkarthick/dialogsum")
    parser.add_argument("--output_dir", type=str, default="fine-tuned-flant5")
    parser.add_argument("--overwrite_output_dir", type=bool, default=False)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--evaluation_strategy", type=str, default="no")
    parser.add_argument("--logging_strategy", type=str, default="steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--save_strategy", type=str, default="no")
    parser.add_argument("--hub_strategy", type=str, default="every_save")
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--hub_model_id", type=str, default="fine-tuned-flant5")
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--run_name", type=str, default="flan-t5-base-model")
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    parser.add_argument("--load_best_model_at_end", type=bool, default=False)
    args = parser.parse_args()
    return args

def load_training_arguments(args):
    """
    Tùy thuộc vào việc có cung cấp đường dẫn file cấu hình hay không, hàm này sẽ tải các tham số huấn 
    luyện từ file cấu hình YAML hoặc trực tiếp từ các tham số dòng lệnh đã phân tích. Các tham số này 
    được sử dụng để thiết lập Seq2SeqTrainingArguments, cấu hình chi tiết như kích thước lô, tốc độ 
    học, chiến lược ghi nhật ký và lưu trữ, v.v.
    """
    try:
        if args.configpath is not None:
            config = load_config(configpath=args.configpath)
            training_args = Seq2SeqTrainingArguments(**config["training_args"])

        else:
            training_args = Seq2SeqTrainingArguments(
                output_dir=args.output_dir,
                overwrite_output_dir=args.overwrite_output_dir,
                num_train_epochs=args.num_train_epochs,
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                weight_decay=args.weight_decay,
                evaluation_strategy=args.evaluation_strategy,
                logging_strategy=args.logging_strategy,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                save_strategy=args.save_strategy,
                hub_strategy=args.hub_strategy,
                logging_steps=args.logging_steps,
                save_total_limit=args.save_total_limit,
                learning_rate=args.learning_rate,
                push_to_hub=args.push_to_hub,
                report_to=args.report_to,
                metric_for_best_model=args.metric_for_best_model,
                load_best_model_at_end=args.load_best_model_at_end,
                run_name=args.run_name
            )

        return training_args
    
    except Exception as e:
        logger.error(f"Error while loading training arguments: {e}")
        raise e

def load_callbacks(args) -> list:
    """
    Thiết lập một EarlyStoppingCallback dựa trên các giá trị kiên nhẫn và ngưỡng đã chỉ định. Callback 
    này sẽ dừng huấn luyện sớm nếu chỉ số đã chọn (như lỗi đánh giá) không cải thiện quá một ngưỡng 
    nhất định trong một số lần kiểm tra đánh giá.
    """
    try:
        callbacks = []
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )
        callbacks.append(early_stopping_callback)
        return callbacks
    
    except Exception as e:
        logger.error(f"Error while loading callbacks: {e}")
        raise e

def load_trainer(model, training_args, dataset, tokenizer, args, optimizer):
    try:
        # callbacks = load_callbacks(args)
        # def custom_compute_metrics(eval_preds):
        #     return compute_metrics(eval_preds, tokenizer)

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            optimizer = optimizer
            # callbacks=callbacks,
            # compute_metrics=custom_compute_metrics
        )
        return trainer
    
    except Exception as e:
        logger.error(f"Error while loading trainer: {e}")
        raise e

def load_config(configpath):
    """
    Tải một file cấu hình YAML nếu tồn tại. File này có thể chứa bất kỳ số lượng cài đặt nào cần thiết 
    cho việc huấn luyện, như các tham số mô hình, tham số huấn luyện, hoặc các cấu hình khác.
    """
    if os.path.exists(configpath):
        with open(configpath, "r") as f:
            config = yaml.safe_load(f)
        return config
    
    else:
        return None

if __name__=='__main__':
    args = parse_args()
    print(args)