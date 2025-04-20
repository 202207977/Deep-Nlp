import json

import os
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from typing import List, Dict, Tuple
import torch
from torch.utils.data import DataLoader

# Librerías propias
from src.data.dataloader import AlertDataset


class AlertGenerator:
    def __init__(self, tag2idx: Dict[str, int], model_path='models/t5_alert_model', train_data_path: str = "data/artificial_train_data_alert.json", num_epochs: int = 10) -> None:
        
        """
        Initializes the AlertGenerator class.

        Args:
            tag2idx: dictionary that maps NER tags to indices.
            model_path: path to the pre-trained T5 model and tokenizer.
                Used to load the alert generation model.

        Returns:
            None
        """

        # Check if the model path exists
        if not os.path.exists(model_path):
            print(f"Model path '{model_path}' does not exist. Starting training...")
            self.finetune_alert_model(model_path, train_data_path, num_epochs)  # If the model path doesn't exist, start fine-tuning

        # Load tokenizer and model from the specified path
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

        # Create a zero-shot classification pipeline
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.labels: List[str] = ["Technology", "Finance", "Health", "Politics", "Sports", "Education", "Business", "Entertainment"]
        
        # Load tokenizer and model from the specified path
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

        # Sentiment index to label mapping
        self.sentiment_map: Dict[int, str] = {0: 'negative', 1: 'positive'}
        
        # Index to NER tag mapping
        self.idx2tag: Dict[int, str] = {idx: tag for tag, idx in tag2idx.items()}


    def generate_alerts(self, list_tokens: List[List[str]], ner_outputs: List[List[int]], sa_outputs: List[torch.Tensor], confidences: List[float]) -> List[str]:
        """
        Generates alerts for a batch of inputs using the T5 model.

        Args:
            tokens: list of token sequences. [batch_size, sequence_length].
            ner_outputs: list of NER label sequences. [batch_size, sequence_length].
            sa_outputs: list of sentiment predictions (tensor with a scalar: 0 or 1).
            confidences: list of confidence scores for the sentiment predictions.

        Returns:
            List of generated alert strings. [batch_size].
        """
        # Join tokens for each text
        texts = [" ".join(tokens) for tokens in list_tokens]
    
        # Classify topics in batch
        topic_results = self.classifier(texts, candidate_labels=self.labels)
    
        # Extract the most probable topic for each text
        topics = [result["labels"][0] for result in topic_results]
    
        # Step 4: Generate alerts in a loop (using calculated topics)
        return [
            self.generate_one_alert(tokens, ner_output, sa_output, confidence, topic)
            for tokens, ner_output, sa_output, confidence, topic in zip(list_tokens, ner_outputs, sa_outputs, confidences, topics)
        ]


    def generate_one_alert(self, tokens: List[str], ner_output: List[int], sa_output: torch.Tensor, confidence: float, topic: str) -> str:
        
        """
        Generates a single alert string based on one set of inputs.

        Args:
            tokens: list of input tokens (words). [sequence_length].
            ner_output: list of NER label indices corresponding to each token. [sequence_length].
            sa_output: tensor with a single sentiment prediction value (0 = negative, 1 = positive). [scalar tensor]
            confidence: float confidence score for the sentiment prediction.
            topic: The classified topic for the given text.

        Returns:
            Generated alert string.
        """

        # Extract relevant (token, entity) pairs
        ner_tuples: List[Tuple[str, str]] = self.extract_ner_tuples(tokens, ner_output)

        # Convert sentiment index to label
        sentiment_label: str = self.sentiment_map[sa_output.item()]

        # Format input for T5 model
        input_text: str = (
            f"NER output: {ner_tuples} "
            f"SA output: {{'sentiment': '{sentiment_label}', 'confidence': {confidence:.3f}}} "
            f"Topic: {topic} "
            "Generate an alert based on this information:"
        )

        # Generate alert
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        generated_ids = self.model.generate(input_ids, max_length=40)
        alert: str = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return alert


    def extract_ner_tuples(self, tokens: list, labels: list) -> list:
        """
        Extracts (token, label) pairs from input tokens where the label is not 'O'.

        Args:
            tokens: list of token strings. [sequence_length].
            labels: list of label indices corresponding to the tokens. [sequence_length].

        Returns:
            List of (token, tag) tuples excluding 'O' tags.
        """

        return [
            (token, self.idx2tag[label]) 
            for token, label in zip(tokens, labels) 
            if self.idx2tag[label] != 'O'
        ]


    def finetune_alert_model(self, model_path: str, train_data_path: str = "data/artificial_train_data_alert.json", num_epochs: int = 10) -> None:
        
        """
        Finetunes the alert model if it does not exist. This includes tokenizing the input/output and
        training the model.

        Args:
            model_path: path to save the trained model.
            train_data_path: path to the training data.
            num_epochs: number of epochs for training.

        Returns:
            None

        """

        # Load the pre-trained model and tokenizer
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        tokenizer = T5Tokenizer.from_pretrained('t5-small')

        #  Load training examples from JSON file
        with open(train_data_path, 'r') as file:
            train_examples = json.load(file) 

        # Tokenize input and output data
        inputs = tokenizer([ex["input"] for ex in train_examples], return_tensors="pt", padding=True, truncation=True)
        outputs = tokenizer([ex["output"] for ex in train_examples], return_tensors="pt", padding=True, truncation=True)
        
        # Ignore padding tokens in loss calculation
        outputs['input_ids'][outputs['input_ids'] == tokenizer.pad_token_id] = -100

        # Create the DataLoader
        dataset = AlertDataset(inputs, outputs)
        dataloader = DataLoader(dataset, batch_size=2)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

        # Training loop
        model.train()
        for epoch in range(num_epochs):  
            for batch in dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                loss = outputs.loss

                # Backward pass y optimización
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")


        model.eval()

        # Save the trained model and tokenizer
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        print("Model trained and saved successfully.")

        return None


if __name__ == "__main__":
    tag2idx = {'B-LOC': 0, 'B-MISC': 1, 'B-ORG': 2, 'B-PER': 3, 'I-LOC': 4, 'I-MISC': 5, 'I-ORG': 6, 'I-PER': 7, 'O': 8}
    alert_generator = AlertGenerator(tag2idx)