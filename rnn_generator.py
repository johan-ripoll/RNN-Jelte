import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime

# Enable eager execution globally
tf.config.run_functions_eagerly(True)
# Enable eager execution for tf.data functions
tf.data.experimental.enable_debug_mode()

# Sample data loading function (you can adapt this according to your dataset)
def load_text_files(file_paths, read_percentage=0.8):
    texts = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            full_text = file.read()
            text_length = len(full_text)
            read_length = int(text_length * read_percentage)
            text = full_text[:read_length]
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

# Print the current timestamp at the beginning
start_time = datetime.now()
print(f"Started at: {start_time}")

# Replace this with your data loading logic (list of file paths)
fluent_folder = "input/fluent"
other_folder = "input/not_fluent"

fluent_files = [os.path.join(fluent_folder, file) for file in os.listdir(
    fluent_folder) if file.endswith('.txt')]
other_files = [os.path.join(other_folder, file) for file in os.listdir(
    other_folder) if file.endswith('.txt')]

# Adapt read_percentage on how much of the files should be read
fluent_texts = load_text_files(fluent_files, read_percentage=0.8)
other_texts = load_text_files(other_files, read_percentage=0.8)

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

    # Set batch size and create generator
    batch_size = 32
    train_generator = data_generator(padded_sequences, labels, batch_size)

	# Custom training loop
    epochs = 1 # from 10 to 1
    steps_per_epoch = len(texts) // batch_size
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step in range(steps_per_epoch):
            batch_texts, batch_labels = next(train_generator)
            loss, accuracy = model.train_on_batch(batch_texts, batch_labels)
            print(f"Step {step+1}/{steps_per_epoch} - Loss: {loss} - Accuracy: {accuracy}")
			
			# Manual cleanup to release memory
            del batch_texts, batch_labels
            tf.keras.backend.clear_session()

	# Training finished - Test step below
    end_time = datetime.now()
    print(f"Training ended at: {end_time}")

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

test_end_time = datetime.now()
print(f"Finished at: {test_end_time}")
elapsed_time = test_end_time - start_time
print(f"Total execution time: {elapsed_time}")
