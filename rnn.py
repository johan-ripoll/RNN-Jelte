import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data loading function (you can adapt this according to your dataset)
def load_text_files(file_paths):
    texts = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            texts.append(text)
    return texts

# Replace this with your data loading logic (list of file paths)
fluent_folder = "input/fluent"
other_folder = "input/not_fluent"

fluent_files = [os.path.join(fluent_folder, file) for file in os.listdir(fluent_folder) if file.endswith('.txt')]
other_files = [os.path.join(other_folder, file) for file in os.listdir(other_folder) if file.endswith('.txt')]

fluent_texts = load_text_files(fluent_files)
other_texts = load_text_files(other_files)

# Checking for training data presence
if fluent_texts or other_texts:
    # Assigning labels automatically based on the split of files
    fluent_labels = np.ones(len(fluent_texts))  # Label 1 for fluent Dutch
    other_labels = np.zeros(len(other_texts))   # Label 0 for other texts

    # Concatenate texts and labels
    texts = fluent_texts + other_texts
    labels = np.concatenate([fluent_labels, other_labels])

    # Tokenization and sequence padding
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_sequence_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # Create the RNN model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_sequence_length))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(padded_sequences, labels, epochs=10, batch_size=32, validation_split=0.2)

    # Load text files from the "test" subfolder
    test_folder = "test"
    test_files = [os.path.join(test_folder, file) for file in os.listdir(test_folder) if file.endswith('.txt')]

    for file_path in test_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            new_text = file.read()

            new_sequence = tokenizer.texts_to_sequences([new_text])
            new_padded_sequence = pad_sequences(new_sequence, maxlen=max_sequence_length)
            prediction = model.predict(new_padded_sequence)

            if prediction >= 0.5:
                print(f"The text in '{file_path}' is predicted to be fluent Dutch.")
            else:
                print(f"The text in '{file_path}' is not predicted to be fluent Dutch.")
else:
    print("No text files for training. Cannot make predictions.")
