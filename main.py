import os
import sys

def check_requirements():
    try:
        import numpy
        import pandas
        import nltk
        import tensorflow
        import sklearn
        import tkinter
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages using:")
        print("pip install numpy pandas nltk tensorflow scikit-learn")
        return False

if __name__ == "__main__":
    if check_requirements():
        # Import the launcher
        from app_launcher import main
        main()
    else:
        print("Please install the required packages and try again.")