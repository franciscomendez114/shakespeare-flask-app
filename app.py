from flask import Flask, render_template, request, jsonify
import sys
import os
import re
import time

# Add the model_specs directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'model_specs'))

# Import the model and generate function
from model_specs.model import generate_text, get_tokenization_info

app = Flask(__name__)

def format_generated_text(text):
    """Format the generated text to handle newlines and other formatting tokens"""
    if not text:
        return ""
    
    # Trim leading/trailing whitespace from every line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    
    # Convert newlines to HTML line breaks
    text = text.replace('\n', '<br>')
    
    # Convert multiple spaces to non-breaking spaces for proper formatting
    text = re.sub(r'  +', lambda m: '&nbsp;' * len(m.group()), text)
    
    # Handle other common formatting tokens if they exist
    # You can add more token replacements here as needed
    
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    generated_text = None
    if request.method == 'POST':
        initial_text = request.form.get('initial_text', '')
        num_tokens = int(request.form.get('num_tokens', 50))
        temperature = float(request.form.get('temperature', 1.0))
        top_k = int(request.form.get('top_k', 200))
        
        try:
            # Use the model to generate text and tokens
            raw_text, tokens = generate_text(
                prompt=initial_text,
                max_new_tokens=num_tokens,
                temperature=temperature,
                top_k=top_k
            )
            # Debug: print tokens to console
            print('Generated tokens:', tokens)
            
            # Format the generated text
            generated_text = format_generated_text(raw_text)
        except Exception as e:
            generated_text = f"Error generating text: {str(e)}"
    
    return render_template('index.html', generated_text=generated_text)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/generate_stream', methods=['POST'])
def generate_stream():
    """Endpoint for streaming text generation"""
    if not request.json:
        return jsonify({'success': False, 'error': 'No JSON data provided'})
    
    initial_text = request.json.get('initial_text', '')
    num_tokens = int(request.json.get('num_tokens', 50))
    temperature = float(request.json.get('temperature', 1.0))
    top_k = int(request.json.get('top_k', 200))
    
    try:
        # Generate the text and tokens
        raw_text, tokens = generate_text(
            prompt=initial_text,
            max_new_tokens=num_tokens,
            temperature=temperature,
            top_k=top_k
        )
        # Debug: print tokens to console
        print('Generated tokens:', tokens)
        
        # Format the text
        formatted_text = format_generated_text(raw_text)
        
        return jsonify({
            'success': True,
            'text': formatted_text
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/tokenize', methods=['POST'])
def tokenize_text():
    """Endpoint for getting tokenization information for input text"""
    if not request.json:
        return jsonify({'success': False, 'error': 'No JSON data provided'})
    
    text = request.json.get('text', '')
    
    try:
        tokenization_info = get_tokenization_info(text)
        return jsonify({
            'success': True,
            'tokenization_info': tokenization_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(port=5050, debug=True) 
