import logging

import os
import sys
import torch
from peft import LoraConfig, get_peft_model, TaskType
import argparse # module giúp phân tích các tham số dòng lệnh, cho phép
                # người dùng tùy chỉnh các biến khi chạy script từ terminal

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from utils import *

from model.models import load_model
from data.preprocessing import preprocessing_data
from data.ingest_data import ingest_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def training_pipeline(args: argparse.Namespace):
    try:
        # Load model from checkpoint
        model = load_model(args.checkpoint)
        logger.info("Complete loading model!")

        # Load data from datapath
        data = ingest_data(args.datapath)
        logger.info("Complete loading dataset!")

        # Pre-processing data
        data = preprocessing_data(data, model.tokenizer)
        logger.info("Complete pre-processing dataset!")

        # optimizer LoRA
        lora_config = LoraConfig(
            r = 32,
            lora_alpha = 32,
            target_modules = ["q", "v"],
            lora_dropout = 0.05,
            bias = "none",
            task_type = TaskType.SEQ_2_SEQ_LM
        )
        peft_model = get_peft_model(model.base_model,lora_config)
        # Load training arguments
        training_args = load_training_arguments(args)
        logger.info("Complete loading training arguments!")

        # Load trainer
        trainer = load_trainer(
                            model=peft_model,
                            training_args=training_args,
                            dataset=data,
                            tokenizer=model.tokenizer,
                            args=args
                            )
        logger.info("Complete loading trainer!")

        # Train model
        trainer.train()
        logger.info("Complete training!")

        # Push to Huggingface Hub
        trainer.push_to_hub()
        logger.info("Complete pushing model to hub!")

    except Exception as e:
        logger.error(f"Error while training: {e}")
        raise e
    