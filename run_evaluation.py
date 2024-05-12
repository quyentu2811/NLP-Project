
import warnings
warnings.filterwarnings("ignore")

import logging

from datasets import load_dataset

import os
import sys

import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, path)

from src.evaluate.evaluation import evaluation_rouge
from peft import PeftModel, PeftConfig
import torch

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Evaluation metric")
    parser.add_argument("--datapath", type=str, default="knkarthick/dialogsum")
    parser.add_argument("--checkpoint", type=str, default="google/flan-t5-base")
    args = parser.parse_args()
    
    datapath = args.datapath
    checkpoint = args.checkpoint

    logger.info("Parse arguments!")
    try:

        data = load_dataset(datapath, split="test")
        logger.info(f"Loaded dataset test from: {datapath}")
        
        from src.model.models import GeneralModel
        model = PeftModel.from_pretrained(model=GeneralModel(checkpoint), 
                                    peft_model_id="tuquyennnn/LoRA-FlanT5-small-v1",
                                    device_map={"cuda": 0} if torch.cuda.is_available() else {"cpu": 0})
        logger.info(f"Loaded model from: {checkpoint}")

        results = evaluation_rouge(model, data)

        logger.info(results)
    except Exception as e:
        logger.error(f"Error: {e}")