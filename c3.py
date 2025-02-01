import cv2
import mediapipe as mp
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox, simpledialog
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# Directory to save gesture data
if not os.path.exists('gesture_data'):
    os.makedirs('gesture_data')

# Mediapipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Tkinter GUI
root = tk.Tk()
root.title("Sign Language Trainer & Recognizer")
root.geometry("400x400")  # Increased height for heading space
root.configure(bg='black')  # Set background color

# Add a heading label
heading_label = tk.Label(root, text="REAL TIME ISL TO TEXT TRANSLATOR", font=("Arial", 16, "bold"), fg="#39FF14", bg='black')
heading_label.pack(pady=(20, 10))  # Padding for spacing

# Train button action
def start_training():
    label = simpledialog.askstring("Input", "Enter the label for the gesture:")
    
    if label:
        capture_hand_data(label)
        messagebox.showinfo("Success", f"Gesture data for '{label}' has been captured!")

def capture_hand_data(label):
    cap = cv2.VideoCapture(0)
    data = []
    
    while True:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                data.append(np.array(landmarks).flatten())

        cv2.imshow("Capture Hand Data - Press 'q' to stop", frame)
        
        # Break the loop on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Save the data as a .npy file
    data = np.array(data)
    np.save(f'gesture_data/{label}.npy', data)
    print(f"Data saved for label: {label}")

# Train Model
def train_model():
    data = []
    labels = []
    gesture_data_dir = 'gesture_data'

    for file in os.listdir(gesture_data_dir):
        if file.endswith('.npy'):
            gesture = np.load(os.path.join(gesture_data_dir, file))
            label = file.split('.')[0]  # Label is the filename without extension
            data.append(gesture)
            labels.extend([label] * len(gesture))

    # Convert data and labels to numpy arrays
    data = np.concatenate(data, axis=0)
    labels = np.array(labels)

    # Encode labels as integers
    unique_labels = np.unique(labels)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    y = np.array([label_to_int[label] for label in labels])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

    # Build the model
    model = models.Sequential([
        layers.Input(shape=(63,)),  # 21 landmarks with x, y, z coordinates
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(unique_labels), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Save the trained model
    model.save('sign_language_custom_model.h5')
    print("Model trained and saved as 'sign_language_custom_model.h5'")
    messagebox.showinfo("Training Complete", "Model has been trained and saved!")

# Recognize Gesture in Real Time
def recognize_gesture():
    cap = cv2.VideoCapture(0)

    # Load the trained model
    try:
        model = load_model('sign_language_custom_model.h5')
    except:
        messagebox.showerror("Error", "Please train the model first!")
        return

    # Load label mapping
    gesture_data_dir = 'gesture_data'
    unique_labels = [file.split('.')[0] for file in os.listdir(gesture_data_dir) if file.endswith('.npy')]
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}

    def predict_gesture(landmarks):
        landmarks = np.array(landmarks).flatten().reshape(1, -1)
        predictions = model.predict(landmarks)
        predicted_label = int_to_label[np.argmax(predictions)]
        return predicted_label

    gesture_buffer = []  # List to store recognized gestures

    while True:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract the landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])

                # Predict the gesture and display it
                gesture = predict_gesture(landmarks)

                # Add gesture to buffer (prevent duplicate entries)
                if not gesture_buffer or gesture_buffer[-1] != gesture:
                    gesture_buffer.append(gesture)

                # Limit the buffer to the last 6 gestures
                if len(gesture_buffer) > 6:
                    gesture_buffer.pop(0)

                # Display the recognized gestures
                cv2.putText(frame, f'Gestures: {" ".join(gesture_buffer)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Real-Time Gesture Recognition", frame)
        
        # Break the loop on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Styling for buttons
def create_button(text, command):
    button = tk.Button(root, text=text, command=command, bg='white', fg='black', font=("Arial", 12, "bold"))
    button.pack(pady=10)
    
    # Button hover effect
    def on_enter(e):
        button['bg'] = "#39FF14"  # Neon green color on hover
    
    def on_leave(e):
        button['bg'] = 'white'  # Back to original color
    
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)
    return button

# Tkinter Buttons
create_button("Capture & Train Gesture", start_training)
create_button("Train Model", train_model)
create_button("Recognize Gesture", recognize_gesture)

# Run Tkinter
root.mainloop()
