import torch
from transformers import LayoutLMForTokenClassification, TrainingArguments, Trainer

encoded_dataset = torch.load('encoded_dataset.pt')
# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_labels = 2  # TODO: Adjut this
model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=num_labels).to(device)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Adjust based on your GPU memory
    per_device_eval_batch_size=2,  # Adjust based on your GPU memory
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    save_steps=500,
)

Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation'],
).train()