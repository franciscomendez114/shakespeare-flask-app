# Shakespeare Flask App

A Flask web application that generates Shakespeare-style text using a custom-trained transformer model. The app provides an interactive web interface for text generation with configurable parameters.

## Features

- **Interactive Text Generation**: Generate Shakespeare-style text from user prompts
- **Configurable Parameters**: Adjust temperature, top-k sampling, and token count
- **Real-time Generation**: Stream text generation with AJAX requests
- **Token Visualization**: View tokenization information for input text
- **Modern Web Interface**: Clean, responsive UI built with HTML/CSS/JavaScript

## Project Structure

```
Shakespeare Flask App/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore           # Git ignore rules
├── templates/            # HTML templates
│   ├── index.html       # Main page template
│   └── about.html       # About page template
└── model_specs/         # Model-related files
    ├── model.py         # Custom transformer model implementation
    ├── model.pth        # Trained model weights
    └── customTokenizer.json  # Custom BPE tokenizer
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Shakespeare-Flask-App
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the application**:
   Open your browser and navigate to `http://localhost:5050`

## Usage

### Text Generation
1. Enter your initial text prompt in the text area
2. Adjust the generation parameters:
   - **Number of Tokens**: How many tokens to generate (default: 50)
   - **Temperature**: Controls randomness (0.1-2.0, default: 1.0)
   - **Top-K**: Number of top tokens to consider (default: 200)
3. Click "Generate Text" to create Shakespeare-style text

### Advanced Features
- **Streaming Generation**: Use the "Generate Stream" button for real-time text generation
- **Token Analysis**: Click "Tokenize Text" to see how your input is tokenized
- **About Page**: Learn more about the project and model architecture

## Model Architecture

The application uses a custom transformer model with the following specifications:

- **Architecture**: Transformer with self-attention
- **Embedding Dimension**: 256
- **Number of Heads**: 4
- **Number of Layers**: 2
- **Context Size**: 256 tokens
- **Vocabulary**: Custom BPE tokenizer trained on Shakespeare text

## API Endpoints

- `GET /` - Main page with text generation form
- `POST /` - Generate text with form data
- `POST /generate_stream` - Stream text generation (JSON)
- `POST /tokenize` - Get tokenization info (JSON)
- `GET /about` - About page

## Dependencies

- Flask - Web framework
- PyTorch - Deep learning framework
- tqdm - Progress bars
- Other standard Python libraries

## Development

The model was trained on Shakespeare's complete works and uses a custom BPE (Byte Pair Encoding) tokenizer. The transformer architecture is implemented from scratch in PyTorch.

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues and enhancement requests! 