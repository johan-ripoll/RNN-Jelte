import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Enable eager execution globally
tf.config.run_functions_eagerly(True)
# Enable eager execution for tf.data functions
tf.data.experimental.enable_debug_mode()

# Sample data loading function (you can adapt this according to your dataset)
def load_text_files(file_paths):
    texts = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            texts.append(text)
    return texts

# Example generator to yield batches of data
def data_generator(texts, labels, batch_size):
    num_samples = len(texts)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_texts = padded_sequences[batch_indices]
            batch_labels = labels[batch_indices]
            yield batch_texts, batch_labels

# Example generator to yield chunks of data
def chunk_data_generator(texts, labels, chunk_size):
    num_samples = len(texts)
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    for chunk_index in range(num_chunks):
        start_idx = chunk_index * chunk_size
        end_idx = min((chunk_index + 1) * chunk_size, num_samples)
        
        chunk_texts = padded_sequences[start_idx:end_idx]
        chunk_labels = labels[start_idx:end_idx]
        
        yield chunk_texts, chunk_labels

# Replace this with your data loading logic (list of file paths)
fluent_folder = "input/fluent"
other_folder = "input/not_fluent"

fluent_files = [os.path.join(fluent_folder, file) for file in os.listdir(
    fluent_folder) if file.endswith('.txt')]
other_files = [os.path.join(other_folder, file) for file in os.listdir(
    other_folder) if file.endswith('.txt')]

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
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1,
              output_dim=128, input_length=max_sequence_length))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'], run_eagerly=True)

    # Set parameters
    epochs = 1
    chunk_size = 5000
    
    # Example usage in the training loop
    train_chunk_generator = chunk_data_generator(texts, labels, chunk_size)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for chunk_texts, chunk_labels in train_chunk_generator:
            loss, accuracy = model.train_on_batch(chunk_texts, chunk_labels)
            print(f"Loss: {loss} - Accuracy: {accuracy}")
            
            # Manual cleanup to release memory
            del chunk_texts, chunk_labels
            tf.keras.backend.clear_session()

    # Load text files from the "test" subfolder
    test_folder = "test"
    test_files = [os.path.join(test_folder, file) for file in os.listdir(
        test_folder) if file.endswith('.txt')]

    for file_path in test_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            new_text = file.read()

            new_sequence = tokenizer.texts_to_sequences([new_text])
            new_padded_sequence = pad_sequences(
                new_sequence, maxlen=max_sequence_length)
            prediction = model.predict(new_padded_sequence)

            if prediction >= 0.5:
                print(
                    f"The text in '{file_path}' is predicted to be fluent Dutch.")
            else:
                print(
                    f"The text in '{file_path}' is not predicted to be fluent Dutch.")
else:
    print("No text files for training. Cannot make predictions.")
