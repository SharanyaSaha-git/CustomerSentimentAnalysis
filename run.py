# run.py  ← place this in your project ROOT
import sys
import os

# Makes sure Python can find the src/ package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.api import app

if __name__ == '__main__':
    app.run(debug=True, port=5000)