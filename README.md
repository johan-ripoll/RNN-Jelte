# Dutch Language Fluency Prediction

This Python script uses a Recurrent Neural Network (RNN) to predict the fluency of Dutch language texts. The RNN model is trained on text data categorized as either "fluent" Dutch or other types of texts.

## Requirements

- Python 3.11 (TensorFlow doesn't support a newer version)
- TensorFlow 2.x - <https://www.tensorflow.org/install/pip#package_location>
- numpy

Install the required libraries using `pip`:

```bash
pip install "tensorflow==2.12.0" numpy
```

## Usage

### Training the Model

1. Organize the training data:

- Place Dutch language text files categorized as "fluent" in the `input/fluent` folder.
- Place other text files in the `input/not_fluent` folder.
- Run the script:

  ```bash
  python rnn.py
  ```

### Testing the Model

1. Organize the test data:

   - Place the text files to be tested in the `test` folder.

2. Run the script:

   ```bash
   python rnn.py
   ```

The script will load the training data, train the RNN model, and then predict the fluency of text files in the `test` folder.

## File Structure

- `rnn.py`: Main Python script to train the RNN model and predict text fluency.
- `input/fluent/`: Folder containing Dutch language text files categorized as "fluent".
- `input/not_fluent/`: Folder containing other text files.
- `test/`: Folder containing text files to test the trained model.

## Notes

- Ensure the text files are in `.txt` format.
- Adjust file paths or modify the script as needed based on your directory structure.
