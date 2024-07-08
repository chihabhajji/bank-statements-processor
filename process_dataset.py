from datasets import load_dataset
from transformers import DetrImageProcessor, AutoTokenizer
import torch
from PIL import Image
import json

ds = load_dataset("tusharshah2006/bank_statements_transactions")

image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
tokenizer = AutoTokenizer.from_pretrained("tusharshah2006/bank_statements_transactions_model")

def preprocess_function(examples):
    all_images = []
    all_texts = []

    for image, gt_str in zip(examples['image'], examples['ground_truth']):
        encoding = image_processor(images=image, return_tensors="pt")
        all_images.append(encoding['pixel_values'])

        gt = json.loads(gt_str)
        gt_entries = gt['gt_parse']['bank_stmt_entries']
        texts = []
        
        for entry in gt_entries:
            for key, value in entry.items():
                if key in ["TXN_DATE", "TXN_DESC", "CHEQUE_REF_NO", "WITHDRAWAL_AMT", "DEPOSIT_AMT", "BALANCE_AMT"]:
                    texts.append(value)
        
        full_text = " ".join(texts)
        all_texts.append(full_text)

    tokenized_texts = tokenizer(all_texts, padding="max_length", truncation=True, return_tensors="pt")
    
    return {
        'pixel_values': torch.cat(all_images),
        'input_ids': tokenized_texts['input_ids'],
        'attention_mask': tokenized_texts['attention_mask']
    }

encoded_dataset = ds.map(preprocess_function, batched=True)
encoded_dataset.set_format(type="torch", columns=["pixel_values", "input_ids", "attention_mask"])

torch.save(encoded_dataset, 'encoded_dataset.pt')