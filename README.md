# PDF to Pinecone Indexing Pipeline

This script provides a robust and efficient way to process a folder of PDF documents, chunk their content, generate embeddings using OpenAI, and upsert them into a Pinecone vector index. It's designed for batch processing and includes progress tracking and error handling.

## Features

- **Batch Processing**: Ingests all PDF files from a specified folder.
- **Token-Aware Chunking**: Uses `tiktoken` to split text into chunks based on token count, preventing errors from exceeding model context limits.
- **Semantic Chunking**: Attempts to split text along sentence and paragraph boundaries for better context preservation.
- **Efficient Upserts**: Batches vector upserts to Pinecone for improved performance.
- **Progress Tracking**: Uses `tqdm` to display a real-time progress bar.
- **Error Handling**: Skips corrupted or un-processable PDFs without halting the entire process.
- **Environment-based Configuration**: Securely manages API keys using a `.env` file.

## Setup

1. **Clone the Repository**
   ```bash
   # (If you created a new repository for this)
   git clone <your-chunker-repo-url>
   cd <chunker-repo-name>
   ```

2. **Create a Virtual Environment**
   It's highly recommended to use a virtual environment to manage dependencies.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   Install all required Python packages from `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` file**
   Copy the example environment file and fill in your credentials.
   ```bash
   cp env.example .env
   ```
   Now, edit the `.env` file with your actual API keys:
   ```dotenv
   # .env
   OPENAI_API_KEY="sk-..."
   PINECONE_API_KEY="your-pinecone-api-key"
   ```

## Usage

Run the `app.py` script from your terminal, providing the path to your folder of PDFs and the name of your target Pinecone index.

**Make sure your Pinecone index has been created with the correct dimension for the embedding model (3072 for `text-embedding-3-large`).**

```bash
python app.py --index_name <your-pinecone-index-name> --pdf_folder <path/to/your/pdf/folder>
```

### Example

If your PDFs are in a folder named `documents` and your Pinecone index is `my-legal-kb`:

```bash
python app.py --index_name my-legal-kb --pdf_folder ./documents
```

The script will start processing the files and display a progress bar.

```
Processing PDFs: 100%|██████████| 194/194 [15:32<00:00, 4.81s/it]
✅ Finished processing all 194 PDF files.
Total chunks created: 25,873
```

## Configuration

The script uses several configurable parameters in the `app.py` file:

- `EMBEDDING_MODEL`: OpenAI embedding model (default: "text-embedding-3-large")
- `CHUNK_SIZE`: Target size of each chunk in tokens (default: 1000)
- `CHUNK_OVERLAP`: Number of tokens to overlap between chunks (default: 100)

## Dependencies

- `openai`: For generating embeddings
- `pinecone`: For vector database operations
- `python-dotenv`: For environment variable management
- `pypdf`: For PDF text extraction
- `tiktoken`: For token-based text chunking
- `tqdm`: For progress bars

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```dotenv
OPENAI_API_KEY="your-openai-api-key"
PINECONE_API_KEY="your-pinecone-api-key"
```

## Error Handling

The script includes robust error handling:
- Skips corrupted or unreadable PDF files
- Handles API rate limits and network errors
- Validates Pinecone index dimensions before processing
- Provides detailed error messages for troubleshooting

## License

This project is licensed under the terms specified in the LICENSE file.
