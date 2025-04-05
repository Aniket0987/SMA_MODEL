import tkinter as tk
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

# Function to save model and tokenizer
def save_model_and_tokenizer(model, tokenizer, model_path='sentiment_model.h5', tokenizer_path='tokenizer.pickle'):
    # Save model
    model.save(model_path)
    
    # Save tokenizer
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Model saved to {model_path}")
    print(f"Tokenizer saved to {tokenizer_path}")

# Main function to run the app
def main():
    root = tk.Tk()
    app = SentimentAnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()