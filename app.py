from flask import Flask, render_template, request, redirect, url_for, send_file
from transformers import pipeline
from rouge import Rouge
from werkzeug.utils import secure_filename
import logging
import os

app = Flask(__name__, template_folder='templates')

# Set up logging
logging.basicConfig(level=logging.INFO)  # Adjust level as needed

# Load the summarization pipeline
summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def summarizer(rawdocs, language="english"):
    try:
        # Perform summarization using the pipeline
        summary_result = summarizer_pipeline(rawdocs, max_length=150, min_length=30, do_sample=False)
        summary = summary_result[0]['summary_text']
        
        # Calculate the number of words in original and summary
        len_orig_txt = len(rawdocs.split(' '))
        len_summary = len(summary.split(' '))
        
        return summary, rawdocs, len_orig_txt, len_summary
    except ValueError as ve:
        logging.error(f"ValueError in summarizer: {str(ve)}")
        raise ve
    except Exception as e:
        logging.error(f"Error in summarizer: {str(e)}")
        raise ValueError("Error in summarizer")

def calculate_rouge(summary, rawdocs):
    rouge = Rouge()
    scores = rouge.get_scores(summary, rawdocs)
    return scores[0]['rouge-1']['f']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/testcases')
def testcases():
    return render_template('testcases.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    rawtext = request.form.get('rawtext')
    language = request.form.get('language')
    
    if not rawtext:
        return render_template('error.html', message="Please provide text to analyze.")
    
    supported_languages = ["english", "hindi", "kannada", "malayalam", "french", "german", "chinese", "korean"]
    
    if language in supported_languages:
        try:
            summary, original_txt, len_orig_txt, len_summary = summarizer(rawtext, language)
            rouge_score = calculate_rouge(summary, rawtext)
            return render_template('summary.html', summary=summary, original_txt=original_txt, len_orig_txt=len_orig_txt, len_summary=len_summary, rouge_score=rouge_score)
        except ValueError as e:
            return render_template('error.html', message=str(e))
    else:
        return render_template('error.html', message="Unsupported language.")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('analyze_uploaded', filename=filename))

@app.route('/analyze_uploaded/<filename>', methods=['GET', 'POST'])
def analyze_uploaded(filename):
    if request.method == 'POST':
        language = request.form.get('language')
    else:
        language = "english"  # Default language
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            rawtext = f.read()
        return render_template('analyze.html', rawtext=rawtext, filename=filename, language=language)
    except Exception as e:
        logging.error(f"Error analyzing uploaded file: {str(e)}")
        return render_template('error.html', message="Error analyzing uploaded file.")

@app.route('/summarize', methods=['POST'])
def summarize():
    rawtext = request.form.get('rawtext')
    language = request.form.get('language')
    filename = request.form.get('filename')
    try:
        summary, original_txt, len_orig_txt, len_summary = summarizer(rawtext, language)
        rouge_score = calculate_rouge(summary, rawtext)
        return render_template('summary.html', summary=summary, original_txt=original_txt, len_orig_txt=len_orig_txt, len_summary=len_summary, rouge_score=rouge_score)
    except ValueError as e:
        return render_template('error.html', message=str(e))
    except Exception as e:
        logging.error(f"Error summarizing: {str(e)}")
        return render_template('error.html', message="Error summarizing.")

@app.route('/download')
def download():
    history_path = os.path.join(app.root_path, 'summarization_history.txt')
    if os.path.exists(history_path):
        return send_file(history_path, as_attachment=True)
    else:
        return render_template('error.html', message="File not found.")

if __name__ == "__main__":
    app.run(debug=True)
