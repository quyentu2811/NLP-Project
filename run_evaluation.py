
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
from peft import PeftModel
import torch
from transformers import AutoModelForSeq2SeqLM

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Evaluation metric")
    parser.add_argument("--datapath", type=str, default="knkarthick/dialogsum")
    parser.add_argument("--checkpoint", type=str, default="google/flan-t5-small")
    parser.add_argument("--modelpath", type=str, default="None")
    args = parser.parse_args()
    
    datapath = args.datapath
    checkpoint = args.checkpoint
    modelpath = args.modelpath

    logger.info("Parse arguments!")

    data = load_dataset(datapath, split="test")
    logger.info(f"Loaded dataset test from: {datapath}")
    
    base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model=base_model, 
                                model_id=modelpath,
                                torch_dtype=torch.bfloat16,
                                is_trainable=False,
                                device_map='auto')
    
    logger.info(f"Loaded model from: {checkpoint}")

    results = evaluation_rouge(model, data)

    logger.info(results)
    print(results)
    # print("results is None")