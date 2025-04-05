import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import sys
import importlib

class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Financial Sentiment Analysis Tool")
        self.root.geometry("600x400")
        self.root.configure(bg="#f0f0f0")
        
        self.create_widgets()
        
    def create_widgets(self):
        # Create a main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Financial Sentiment Analysis Tool", 
                                font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # Description
        desc_text = "This tool helps analyze sentiment in financial news.\n"
        desc_text += "You can train a new model or use an existing one."
        desc_label = ttk.Label(main_frame, text=desc_text, font=("Arial", 10))
        desc_label.pack(pady=10)
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(pady=20)
        
        # Check if model exists
        model_exists = os.path.exists('sentiment_model.h5') and os.path.exists('tokenizer.pickle')
        
        # Launch GUI button
        self.launch_btn = ttk.Button(buttons_frame, text="Launch Analysis GUI", 
                                    command=self.launch_gui,
                                    state="normal" if model_exists else "disabled")
        self.launch_btn.pack(pady=10, ipady=10, ipadx=20)
        
        if not model_exists:
            warning_label = ttk.Label(buttons_frame, text="No model found. Please train a model first.", 
                                    foreground="red")
            warning_label.pack(pady=5)
        
        # Training options
        train_frame = ttk.LabelFrame(main_frame, text="Training Options")
        train_frame.pack(fill=tk.X, pady=10, padx=10)
        
        # Data file selection
        file_frame = ttk.Frame(train_frame)
        file_frame.pack(fill=tk.X, pady=10, padx=10)
        
        ttk.Label(file_frame, text="Data File (optional):").pack(side=tk.LEFT, padx=5)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=30)
        file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        # Train button
        train_btn = ttk.Button(train_frame, text="Train New Model", command=self.train_model)
        train_btn.pack(pady=10, ipady=5, ipadx=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
    
    def train_model(self):
        # Import training module
        from train_model import train_model
        
        data_path = self.file_path_var.get().strip()
        if data_path and not os.path.exists(data_path):
            messagebox.showerror("Error", f"File not found: {data_path}")
            return
        
        self.status_var.set("Training model... Please wait")
        self.root.update()
        
        # Run training in a separate thread to prevent GUI freezing
        def training_thread():
            try:
                train_model(data_path if data_path else None)
                self.status_var.set("Training completed successfully!")
                self.launch_btn.config(state="normal")
                messagebox.showinfo("Success", "Model training completed successfully!")
            except Exception as e:
                self.status_var.set(f"Error during training: {str(e)}")
                messagebox.showerror("Error", f"Training failed: {str(e)}")
        
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()
    
    def launch_gui(self):
        # Import the GUI module
        from sentiment_gui import main as run_gui
        
        # Close the launcher and start the GUI
        self.root.destroy()
        run_gui()

def ensure_modules():
    """Create the necessary modules if they don't exist"""
    # Create train_model.py if it doesn't exist
    if not os.path.exists('train_model.py'):
        with open('train_model.py', 'w') as f:
            f.write("""import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os

def train_model(data_path=None):
    # Download NLTK data
    nltk.download('vader_lexicon')

    # Initialize Sentiment Analyzer
    sia = SentimentIntensityAnalyzer()

    # Example input data with sentiment labels (use this if no data path provided)
    default_data = [
        {
            "title": "Sensex Drops on Weak Global Cues",
            "description": "Weak global cues led to a drop in the Sensex.",
            "sentiment": "negative"
        },
        {
            "title": "Nifty Falls Amid Currency Depreciation",
            "description": "Currency depreciation caused Nifty to fall.",
            "sentiment": "negative"
        },
        {
            "title": "Market Sentiment Dampens on Rising Inflation",
            "description": "Rising inflation dampened market sentiment.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Soars on Positive Earnings",
            "description": "Positive earnings reports pushed the Sensex higher.",
            "sentiment": "positive"
        },
        {
            "title": "Nifty Ends at Record High",
            "description": "Nifty closed at a record high today.",
            "sentiment": "positive"
        },
        {
            "title": "Markets Rally on Rate Cut Hopes",
            "description": "Stock markets rallied on hopes of an interest rate cut.",
            "sentiment": "positive"
        }
    ]

    # Load data or use default
    if data_path and os.path.exists(data_path):
        try:
            # Assuming CSV format with title, description, and sentiment columns
            df = pd.read_csv(data_path)
            data = df.to_dict('records')
            print(f"Loaded {len(data)} records from {data_path}")
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Using default data instead.")
            data = default_data
    else:
        print("No data path provided or file not found. Using default data.")
        data = default_data

    # Preprocess text data
    def preprocess_data(data):
        texts = []
        labels = []
        vader_scores = []
        for item in data:
            text = item['title'] + " " + item['description']
            sentiment = item['sentiment']
            texts.append(text)
            labels.append(sentiment)
            vader_score = sia.polarity_scores(text)['compound']
            vader_scores.append(vader_score)
        return texts, labels, vader_scores

    texts, labels, vader_scores = preprocess_data(data)

    MAX_NUM_WORDS = 10000
    MAX_SEQUENCE_LENGTH = 100
    EMBEDDING_DIM = 100

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data_matrix = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # sentiment labels to numerical values
    def convert_sentiments_to_labels(labels):
        label_map = {'positive': 1, 'negative': 0}
        return [label_map[label] for label in labels]

    labels = convert_sentiments_to_labels(labels)

    X_train, X_test, y_train, y_test, vader_train, vader_test = train_test_split(
        data_matrix, labels, vader_scores, test_size=0.2, random_state=42
    )

    unique_labels = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    class_weight_dict = dict(enumerate(class_weights))

    # Define LSTM model with VADER score input
    text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='text_input')
    embedding_layer = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(text_input)
    lstm_layer = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(embedding_layer)

    vader_input = Input(shape=(1,), dtype='float32', name='vader_input')
    concatenated = Concatenate()([lstm_layer, vader_input])

    dense = Dense(64, activation='relu')(concatenated)
    output = Dense(2, activation='softmax')(dense)

    model = Model(inputs=[text_input, vader_input], outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    # Train the model
    history = model.fit(
        [X_train, np.array(vader_train)],
        np.array(y_train),
        epochs=10,
        batch_size=16,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    score, accuracy = model.evaluate([X_test, np.array(vader_test)], np.array(y_test), verbose=1)
    print(f"Test loss: {score}")
    print(f"Test accuracy: {accuracy}")

    # Save model and tokenizer
    model.save('sentiment_model.h5')
    
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Model and tokenizer saved successfully!")
    
    return model, tokenizer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--data', type=str, help='Path to CSV data file', default=None)
    
    args = parser.parse_args()
    
    train_model(args.data)
""")
    
    # Create sentiment_gui.py if it doesn't exist
    if not os.path.exists('sentiment_gui.py'):
        with open('sentiment_gui.py', 'w') as f:
            f.write("""import tkinter as tk
from tkinter import ttk, scrolledtext
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

class SentimentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Financial News Sentiment Analyzer")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize NLTK
        nltk.download('vader_lexicon', quiet=True)
        self.sia = SentimentIntensityAnalyzer()
        
        # Load model and tokenizer
        try:
            self.model = load_model('sentiment_model.h5')
            with open('tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            self.model_loaded = True
            self.status_message = "Model loaded successfully!"
        except Exception as e:
            self.model_loaded = False
            self.status_message = f"Model not found. Please train the model first. Error: {str(e)}"
        
        # Constants for the model
        self.MAX_SEQUENCE_LENGTH = 100
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Create a main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title label
        title_label = ttk.Label(main_frame, text="Financial News Sentiment Analyzer", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Input Frame
        input_frame = ttk.LabelFrame(main_frame, text="Enter Financial News")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title input
        title_label = ttk.Label(input_frame, text="News Title:")
        title_label.pack(anchor=tk.W, padx=5, pady=5)
        
        self.title_entry = ttk.Entry(input_frame, width=80)
        self.title_entry.pack(fill=tk.X, padx=5, pady=5)
        
        # Description input
        desc_label = ttk.Label(input_frame, text="News Description:")
        desc_label.pack(anchor=tk.W, padx=5, pady=5)
        
        self.desc_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=10)
        self.desc_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        # Analyze button
        self.analyze_btn = ttk.Button(buttons_frame, text="Analyze Sentiment", 
                                    command=self.analyze_sentiment)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        clear_btn = ttk.Button(buttons_frame, text="Clear", command=self.clear_inputs)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Results Frame
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a frame for results display
        display_frame = ttk.Frame(results_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # VADER Score
        ttk.Label(display_frame, text="VADER Score:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.vader_score_var = tk.StringVar()
        ttk.Label(display_frame, textvariable=self.vader_score_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Model Prediction
        ttk.Label(display_frame, text="Model Prediction:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.prediction_var = tk.StringVar()
        self.prediction_label = ttk.Label(display_frame, textvariable=self.prediction_var, font=("Arial", 12, "bold"))
        self.prediction_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Confidence Score
        ttk.Label(display_frame, text="Confidence:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.confidence_var = tk.StringVar()
        ttk.Label(display_frame, textvariable=self.confidence_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set(self.status_message)
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Set style for prediction label
        self.update_prediction_display("")
        
    def analyze_sentiment(self):
        if not self.model_loaded:
            self.status_var.set("Model not loaded. Please train the model first.")
            return
        
        title = self.title_entry.get().strip()
        description = self.desc_text.get("1.0", tk.END).strip()
        
        if not title and not description:
            self.status_var.set("Please enter either a title or description.")
            return
            
        # Combine text
        text = f"{title} {description}"
        
        # Get VADER score
        vader_score = self.sia.polarity_scores(text)['compound']
        self.vader_score_var.set(f"{vader_score:.4f}")
        
        # Predict using the model
        try:
            sequence = self.tokenizer.texts_to_sequences([text])
            padded_sequence = pad_sequences(sequence, maxlen=self.MAX_SEQUENCE_LENGTH)
            prediction = self.model.predict([padded_sequence, np.array([vader_score])])
            
            # Get prediction class and confidence
            pred_class = np.argmax(prediction)
            confidence = prediction[0][pred_class]
            
            label_map = {0: 'Negative', 1: 'Positive'}
            sentiment = label_map[pred_class]
            
            # Update UI
            self.prediction_var.set(sentiment)
            self.confidence_var.set(f"{confidence:.2%}")
            self.update_prediction_display(sentiment)
            
            self.status_var.set("Analysis completed successfully!")
        except Exception as e:
            self.status_var.set(f"Error during prediction: {str(e)}")
    
    def update_prediction_display(self, sentiment):
        if sentiment.lower() == "positive":
            self.prediction_label.configure(foreground="green")
        elif sentiment.lower() == "negative":
            self.prediction_label.configure(foreground="red")
        else:
            self.prediction_label.configure(foreground="black")
    
    def clear_inputs(self):
        self.title_entry.delete(0, tk.END)
        self.desc_text.delete("1.0", tk.END)
        self.vader_score_var.set("")
        self.prediction_var.set("")
        self.confidence_var.set("")
        self.update_prediction_display("")
        self.status_var.set("Ready")

def main():
    root = tk.Tk()
    app = SentimentAnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
""")

def main():
    # Create necessary files
    ensure_modules()
    
    # Start the launcher
    root = tk.Tk()
    app = LauncherApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()