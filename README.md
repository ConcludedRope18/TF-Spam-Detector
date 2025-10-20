# Spam Email Classifier with LSTM (TensorFlow)

This project is a machine learning model for classifying emails as spam or ham (not spam). It uses an LSTM neural network trained on a balanced dataset of labeled email samples. The project supports training, saving the model and tokenizer, and classifying new emails from plain text files.

---

## Features

* Preprocesses email text (stopword removal, punctuation cleanup, tokenization)
* Trains an LSTM-based classifier
* Automatically balances the dataset
* Uses early stopping and learning rate adjustment
* Saves trained model and tokenizer
* Loads saved model to classify emails from a folder

---

## Project Structure

```
project/
├── model.py                              # Trains the model and saves it
├── classify.py                           # Loads model/tokenizer and classifies new emails
├── spam_ham_dataset.csv                  # Dataset used for training
├── spam_classifier_model.h5              # Best performing model checkpoint
├── tokenizer.pkl                         # Tokenizer saved for reuse
├── emails/                               # Folder of .txt emails to be classified
```

---

## Installation

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install tensorflow pandas numpy nltk scikit-learn
```

---

## How to Train the Model

```bash
python model.py
```

This script will:

* Load and preprocess the dataset
* Train the LSTM model
* Save the best model as `spam_classifier_model.h5`
* Save the tokenizer as `tokenizer.pkl`

---

## How to Classify New Emails

1. Place your emails as `.txt` files in the `emails` folder.
2. Run the script:

```bash
python classifier.py
```

3. Output will be:

```
email1.txt: Ham (0.12)
email2.txt: Spam (0.91)
```

---

## How It Works

* **Preprocessing**:

  * Lowercasing, removing punctuation and stopwords
  * Tokenizing and padding sequences to a fixed length
* **Model Architecture**:

  * Embedding layer: Learns vector representations of words
  * LSTM Layer: Learns patterns in sequences
  * Hidden Dense Layer: Learns features of patterns from the LSTM Layer
  * Output Dense Layer: Predicts if the email is spam or ham
* **Training Enhancements**:

  * Early stopping
  * ReduceLROnPlateau for dynamic learning rate
  * Checkpoints to save the best version of the model

---

## Configuration

Inside the scripts:

* `MAX_LEN = 100` → Max sequence length for each email
* `EMAIL_FOLDER = 'emails'` → Folder containing new emails to classify

---
