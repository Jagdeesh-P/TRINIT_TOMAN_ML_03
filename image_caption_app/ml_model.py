'''import pandas as pd
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout

# Load the RSICD dataset
train_df = pd.read_csv("cleaned_train.csv", error_bad_lines=False)
test_df = pd.read_csv("cleaned_test.csv", error_bad_lines=False)


# Load and preprocess images
def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    return img

train_images = [load_image(image_path) for image_path in train_df["image"]]
test_images = [load_image(image_path) for image_path in test_df["image"]]

# Tokenize captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df["caption"])
vocab_size = len(tokenizer.word_index) + 1

# Convert captions to sequences of integers
train_sequences = tokenizer.texts_to_sequences(train_df["caption"])
test_sequences = tokenizer.texts_to_sequences(test_df["caption"])

# Pad sequences
max_length = max(len(seq) for seq in train_sequences)
train_sequences = pad_sequences(train_sequences, maxlen=max_length, padding="post")
test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding="post")

# Define the CNN model for feature extraction
resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
resnet.trainable = False
flatten = tf.keras.layers.Flatten()(resnet.output)
cnn_model = Model(inputs=resnet.input, outputs=flatten)

# Define the RNN model for caption generation
inputs = Input(shape=(max_length,))
embedding_layer = Embedding(vocab_size, 300, input_length=max_length)(inputs)
lstm1 = LSTM(256, return_sequences=True)(embedding_layer)
dropout = Dropout(0.5)(lstm1)
lstm2 = LSTM(256)(dropout)
outputs = Dense(vocab_size, activation='softmax')(lstm2)

# Combine the CNN and RNN models
image_input = Input(shape=(224, 224, 3))
features = cnn_model(image_input)
features = tf.expand_dims(features, axis=1)
features = tf.repeat(features, repeats=max_length, axis=1)
combined = tf.concat([features, embedding_layer], axis=-1)
rnn_model = Model(inputs=[image_input, inputs], outputs=outputs)

# Compile the model
rnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model
rnn_model.fit([np.array(train_images), train_sequences], np.expand_dims(train_sequences, axis=-1), epochs=10, batch_size=64)

# Generate captions for input image
def generate_caption(image_path):
    img = load_image(image_path)
    img = np.expand_dims(img, axis=0)
    features = cnn_model.predict(img)
    input_sequence = tokenizer.texts_to_sequences(["<start>"])[0]
    caption = "<start>"

    while True:
        input_sequence = pad_sequences([input_sequence], maxlen=max_length, padding="post")
        output_sequence = rnn_model.predict([img, input_sequence])
        predicted_word_idx = np.argmax(output_sequence[0, -1, :])
        predicted_word = tokenizer.index_word[predicted_word_idx]

        if predicted_word == "<end>" or len(caption.split()) > max_length:
            break

        caption += " " + predicted_word
        input_sequence[0, -1] = predicted_word_idx

    return caption

# Example usage
image_path = "Image.jpg"
caption = generate_caption(image_path)
print("Generated Caption:", caption)'''

def generate_caption(image_path):
    return "....Generating Caption...."