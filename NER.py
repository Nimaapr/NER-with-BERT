# Import necessary libraries
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Function to tokenize the input sentences while preserving the entity labels
def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    # Tokenize each word and assign labels to each sub-word
    for word, label in zip(sentence.split(), text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

# Function to evaluate a sentence
def evaluate(sentence, text_labels):
    tokenized_sentence, labels = tokenize_and_preserve_labels(sentence, text_labels)
    
    # Add special tokens needed for BERT inputs
    tokenized_sentence = ['[CLS]'] + tokenized_sentence + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_sentence)
    attention_mask = [1] * len(input_ids)
    
    # Convert to PyTorch tensors
    input_ids = torch.tensor([input_ids])
    attention_mask = torch.tensor([attention_mask])
    
    # Forward pass through the model to get logits
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs[0]
    # Get the most probable label index for each token
    predictions = torch.argmax(logits, dim=-1).numpy().tolist()

    # Exclude the special tokens from the predictions
    predictions = predictions[0][1:-1]
    
    # Convert the numerical labels back to their string counterparts
    predictions = [model.config.id2label[pred] for pred in predictions]

    # Calculate evaluation metrics
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    return {"precision": precision, "recall": recall, "f1": f1}

# Test sentences and corresponding labels
sentences = [
    "My name is John Doe and I live in New York.",
    "Sarah works at Google in San Francisco.",
    "The Eiffel Tower is in Paris.",
    "I visited London and met with Queen Elizabeth."
]
label_sets = [
    ["O", "O", "O", "B-PER", "I-PER", "O", "O", "O", "O", "B-LOC", "I-LOC"],
    ["B-PER", "O", "O", "B-ORG", "O", "B-LOC", "I-LOC"],
    ["O", "B-LOC", "O", "O", "B-LOC"],
    ["O", "O", "B-LOC", "O", "O", "O", "B-PER", "I-PER"]
]

# Calculate and print average precision, recall, and F1 score over all sentences
precisions = []
recalls = []
f1s = []
for sentence, labels in zip(sentences, label_sets):
    scores = evaluate(sentence, labels)
    precisions.append(scores["precision"])
    recalls.append(scores["recall"])
    f1s.append(scores["f1"])

avg_precision = sum(precisions) / len(precisions)
avg_recall = sum(recalls) / len(recalls)
avg_f1 = sum(f1s) / len(f1s)

print(f"Average precision: {avg_precision}")
print(f"Average recall: {avg_recall}")
print(f"Average F1 score: {avg_f1}")
