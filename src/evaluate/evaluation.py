import logging

import os
import sys

from datasets import Dataset

import evaluate

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from model.models import GeneralModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Config
# from data.ingest_data import ingest_data
# from data.data_strategy import PostPreprocessData


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RougeEvaluation:
    def __init__(self) -> None:
        self.rouge_metric = evaluate.load("rouge")
        
    def compute_rouge_metric(self, generated_summary, reference_summary) -> dict:
        results = self.rouge_metric.compute(
            predictions=generated_summary,
            references=reference_summary,
            use_aggregator=True,
            use_stemmer=True
        )
        return results
    

def evaluation_rouge(model: GeneralModel, data: Dataset) -> dict:
    dialogues = data["dialogue"]

    human_summaries = [summary for summary in data["summary"]]

    model_summaries = []

    tokenizer = AutoTokenizer.from_pretrained(model.base_model.config._name_or_path)

    prefix = "Summarize the following conversation:\n\n"
    suffix = "\n\nSummary: "

    for idx, dialogue in enumerate(dialogues):
        input_text = prefix + dialogue + suffix
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        output_ids = model.generate(input_ids)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        model_summaries.append(output_text)

    rouge_evaluator = RougeEvaluation()

    results = rouge_evaluator.compute_rouge_metric(model_summaries, human_summaries)
    
    generated_lengths = [len(summary.split()) for summary in model_summaries]
    average_gen_len = sum(generated_lengths) / len(generated_lengths) if generated_lengths else 0

    results["gen_len"] = average_gen_len

    return results

if __name__=='__main__':
    pass

# if __name__=='__main__':
#     parser = argparse.ArgumentParser(description="Evaluation metric")
#     parser.add_argument("--datapath", type=str, default="knkarthick/dialogsum")
#     parser.add_argument("--checkpoint", type=str, default="google/flan-t5-base")
#     args = parser.parse_args()


#     datapath = args.datapath
#     checkpoint = args.checkpoint

#     data = load_dataset(datapath, split="test")

#     model = GeneralModel(checkpoint)

#     results = evaluation_rouge(model, data)
#     logger.info(results)
#     print(results)