# Spam Detection Using Deep Learning

This project demonstrates the use of deep learning techniques, specifically LSTM (Long Short-Term Memory), for spam message detection. The model classifies email or message content into two categories: Spam or Ham (non-spam). We use TensorFlow and Keras to build the model and preprocess text data for effective classification.

## Project Structure

- **Main_Model.ipynb**: The main Jupyter notebook for building and training the spam detection model.
- **my_model.keras**: The trained deep learning model saved in Keras format.
- **spam_detection_model.keras**: Another version of the trained deep learning model.
- **tempCodeRunnerFile.py**: A temporary Python script used during development.
- **.ipynb_checkpoints**: Jupyter notebook checkpoints for saving intermediate progress.
- **Dataset**: Directory containing the dataset used for training the model.


## Snap Shots

#### 1. Data Head:

<img src="https://github.com/user-attachments/assets/1368423e-4b2f-4a96-8f73-5567a7fec5ca" width="400">

#### 2. Count of Spam & Downsampling::

<img src="https://github.com/user-attachments/assets/66f306d4-4084-427e-a764-2ba7722ce8f0" width="400">
<img src="https://github.com/user-attachments/assets/510c8bde-12e2-434f-82ee-b71c4654d2fc" width="400">

#### 4. Word cloud to visualize terms in "Spam" and "Non-Spam" words.

<img src="https://github.com/user-attachments/assets/b57f9080-d69c-4576-b119-5dac917ec5ac" width="400">  
<img src="https://github.com/user-attachments/assets/21844ad9-a7f4-4640-afee-512152263452" width="400">

#### 5. Accuracy and Testing Result::

<img src="https://github.com/user-attachments/assets/59b90368-3389-463b-bcad-cfd6b7c59537" width="400">
<img src="https://github.com/user-attachments/assets/e01b66c1-1194-4111-9008-1cc61d3f1de7" width="400">

#### 7. Real Time Testing on Mail ( Spam & Not-Spam ) :
                        
   <img src="https://github.com/user-attachments/assets/1d4391ac-3107-474c-98a7-ec419748f02f" width="400">
   <img src="https://github.com/user-attachments/assets/ed2fcb7f-0714-47b0-becb-5fc631fb790f" width="400">

 

   
## Requirements

Make sure to install the necessary libraries before running the project:

```bash
  pip install tensorflow pandas numpy matplotlib seaborn nltk wordcloud scikit-learn.
```

## Dataset

The dataset used for this project contains labeled email or message data. The two classes are:

- **Spam**: Unsolicited or unwanted messages.
- **Ham**: Non-spam messages.

The dataset is preprocessed to remove stop words, punctuation, and irrelevant text to enhance the model’s performance.

## Steps Involved

### 1. Data Preprocessing
- The dataset is cleaned by removing unwanted text like email headers, subject lines, and punctuation.
- Stopwords are removed to focus on the relevant terms in the messages.
- A word cloud is generated to visualize the most common words in both spam and non-spam emails.

### 2. Tokenization and Padding
- The text is tokenized to convert words into sequences of integers.
- Sequences are padded to ensure uniform input length for the neural network.

### 3. Model Building
A deep learning model is built using TensorFlow/Keras:
- **Embedding Layer**: Converts words into dense vectors.
- **LSTM Layer**: Captures patterns and context in the sequence of words.
- **Dense Layers**: Processes the features and outputs a classification result (Spam or Ham).
- **Activation Function**: Sigmoid for binary classification.

### 4. Training the Model
- The model is trained using the training data, with callbacks to reduce learning rate on plateau and early stopping to prevent overfitting.

### 5. Model Evaluation
- The model's performance is evaluated on the test dataset, achieving a high accuracy of 97.83%.

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/message-detection-using-deep-learning.git
```

2. Open Main_Model.ipynb in Jupyter Notebook or Google Colab and run the cells to build, train, and evaluate the spam detection model.

##  Model Usage
Once the model is trained, you can use the saved .keras model files to make predictions on new messages. Load the model as follows:

```python
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('spam_detection_model.keras')

# Predict whether a message is spam or ham
prediction = model.predict([new_message])
```

## Contributing
Feel free to fork this project and submit pull requests. Contributions are welcome!

## License
This project is licensed under the MIT License.
