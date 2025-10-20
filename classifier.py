import os
import pickle
import string
import tensorflow as tf
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('spam_classifier_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.replace('Subject', '')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

EMAIL_FOLDER = 'emails'
MAX_LEN = 100

for filename in os.listdir(EMAIL_FOLDER):
    if filename.endswith('.txt'):
        path = os.path.join(EMAIL_FOLDER, filename)
        with open(path, 'r', encoding='utf-8') as file:
            raw_text = file.read()

        cleaned_text = preprocess_text(raw_text)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

        prediction = model.predict(padded)[0][0]
        label = 'Spam' if prediction >= 0.5 else 'Ham'

        print(f"{filename}: {label} ({prediction:.2f})")
