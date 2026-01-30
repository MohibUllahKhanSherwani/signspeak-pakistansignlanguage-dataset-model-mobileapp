# train_model.py
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

from actions_config import load_actions, DATA_PATH, SEQUENCE_LENGTH, NUM_SEQUENCES, BATCH_SIZE, EPOCHS, LEARNING_RATE

def load_data(actions):
    sequences, labels = [], []
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            raise FileNotFoundError(f"Data folder missing for action: {action}")
        for seq in range(NUM_SEQUENCES):
            window = []
            for frame_num in range(SEQUENCE_LENGTH):
                npy_path = os.path.join(action_path, str(seq), f"{frame_num}.npy")
                if not os.path.exists(npy_path):
                    raise FileNotFoundError(f"Missing file: {npy_path}. Did you finish data collection?")
                frame = np.load(npy_path)
                window.append(frame)
            sequences.append(window)
            labels.append(action)
    return np.array(sequences), np.array(labels)

def build_model(input_shape, num_classes):
    model = Sequential()
    # LSTM layers (tanh is standard/stable for LSTMs)
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True, activation='tanh'))
    model.add(LSTM(64, return_sequences=False, activation='tanh'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    actions = load_actions()
    print("Loading data...")
    X, y = load_data(actions)
    # shape check
    print("X shape:", X.shape)  # expected: (num_samples, SEQUENCE_LENGTH, features)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_cat = to_categorical(y_encoded).astype(int)
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, y_cat.shape[1])
    model.summary()
    # Train
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    # Save model and label encoder
    model.save("action_model.h5")
    joblib.dump(le, "label_encoder.pkl")
    print("Training complete. Model saved to action_model.h5 and label_encoder.pkl")

if __name__ == "__main__":
    main()
