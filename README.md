# Automatic Alert Generation with NER and SA

## A complete Deep Learning + NLP project using joint models to generate alerts from text. Combines Named Entity Recognition (NER), Sentiment Analysis (SA), and sequence-to-sequence generation.

This is the final project for the **DL+NLP** course. It explores the design, training, and evaluation of a multitask model that performs both NER and SA to produce informative alerts from text data. The alerts can be used in scenarios such as reputation tracking, financial monitoring, and geopolitical risk detection.

The project implements a full pipeline from raw data to alert generation, including joint neural architectures (BigNet, SmallNet) and a custom finetuned alert generator based on T5.

## Reproduction Steps

Follow these steps to reproduce the results and generate your own alerts:

1. **Environment Setup:**
   - Use Python 3.8 or later.
   - Install dependencies with:  
     ```bash
     pip install -r requirements.txt
     ```

2. **Data Preparation:**
   - Convert NER-only data to multitask format with sentiment using:
     ```bash
     python -m src.data.process_nerd
     ```
   - This creates `dataset.csv` with combined `tokens`, `ner_tags`, and `sentiment`.

3. **Training:**
   - Train your model with:
     ```bash
     python -m src.train
     ```
   - By default, this trains `BigNet` using both SA and NER losses jointly, saving the model under `models/`.

4. **Evaluation:**
   - Evaluate the model and (optionally) generate alerts:
     ```bash
     python -m src.evaluate
     ```
   - This script computes SA accuracy and NER F1 score, and shows real-time alert examples.

5. **Alert Generator (Optional Pretraining):**
   - If the T5 model for alert generation does not exist, it will be trained from:
     ```bash
     data/artificial_train_data_alert.json
     ```
    - You can manually trigger training by running:
      ```bash
      python -m src.Mymodels.AlertGenerator
      ```


## Dataset

We use a custom-processed version of the **CoNLL-2003** dataset. We enrich it with synthetic sentiment labels to simulate real-world social media or news streams.

Each example contains:
- A list of tokens (words)
- Named Entity tags in BIO format (`B-ORG`, `I-LOC`, etc.)
- A global sentiment label (`positive`, `negative`, etc.)

Example:
```json
{
  "tokens": ["Tesla", "stocks", "plummet", "amid", "controversy"],
  "ner_tags": ["B-ORG", "O", "O", "O", "O"],
  "sentiment": "negative"
}
```

## Structure

```
|-- ProyectoFinal/
|   |-- data/
|   |   |-- artificial_train_data_alert.json
|   |   |-- dataset.csv
|   
|   |-- models/
|   |   |-- Best_model.pt
|   
|   |-- runs/
|   |   |-- Best_model
|   
|   |-- src/
|       |-- data/
|       |   |-- __init__.py
|       |   |-- dataloader.py
|       |   |-- data_functions.py
|       |   |-- process_nerd.py
|       |-- Mymodels/
|       |   |-- __init__.py
|       |   |-- AlertGenerator.py
|       |   |-- BigNet.py
|       |   |-- SmallNet.py
|       |-- __init__.py
|       |-- evaluate.py
|       |-- train.py
|       |-- train_functions.py
|       |-- utils.py
```

## Modules

### `process_nerd.py`
Transforms a raw NER dataset into one with added sentiment, using a Hugging Face `pipeline("sentiment-analysis")`.

### `train.py`
Full training loop for multitask learning with:
- SA and NER losses
- TensorBoard logging
- Learning rate scheduler
- BigNet and SmallNet architectures

### `evaluate.py`
Evaluates a trained model on validation/test data:
- Computes NER F1 score (`seqeval`)
- Computes SA accuracy
- Optionally generates alerts via the T5 model

### `BigNet.py`
Joint architecture for NER and SA. Combines:
- Word embeddings (Word2Vec)
- BiLSTM for NER
- Character embeddings via Flair
- Final LSTM for sentiment from char-level + NER
- Alert generation based on predicted entities

### `SmallNet.py`
Lightweight dual-stream model with:
- Shared word embeddings
- Two separate LSTMs: one for NER, one for SA

### `AlertGenerator.py`
Alert generation module based on a fine-tuned T5 model:
- Inputs: (NER tags, SA result, topic classification)
- Output: a textual alert, e.g., `"Negative sentiment detected for Hillary Clinton with high confidence in the area of Politics."`
- Includes rule-based data creation and zero-shot topic classification (BART)

## Requirements

```python
torch==2.0.0
transformers==4.40.1
flair==0.12.2
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.1
seqeval==1.2.2
gensim==4.3.2
tqdm==4.66.1
matplotlib==3.8.0
tensorboard==2.14.1
```

## Recommendations

- Use GPU for training BigNet; Flair embeddings are slow on CPU.
- Train all components sequentially: first run `process_nerd.py`, then `train.py`, then `evaluate.py`.
- Save checkpoints often if experimenting with architecture variations.
- Add your own test sentence or post to generate a custom alert.