# PDF to Pinecone Indexing Pipeline

This script provides a robust and efficient way to process a folder of PDF documents, chunk their content, generate embeddings using OpenAI, and upsert them into a Pinecone vector index. It's designed for batch processing and includes progress tracking and error handling.

## Features

- **Batch Processing**: Ingests all PDF files from a specified folder.
- **Token-Aware Chunking**: Uses `tiktoken` to split text into chunks based on token count, preventing errors from exceeding model context limits.
- **Intelligent Batch Embedding**: Automatically batches embedding requests to stay under OpenAI's 300k token limit.
- **Semantic Chunking**: Attempts to split text along sentence and paragraph boundaries for better context preservation.
- **Resume Functionality**: Robust state management allows resuming interrupted processing from where it left off.
- **Metadata Size Management**: Automatically truncates content to stay within Pinecone's 40KB metadata limit.
- **Efficient Upserts**: Batches vector upserts to Pinecone for improved performance.
- **Progress Tracking**: Uses `tqdm` to display a real-time progress bar with file-by-file updates.
- **Error Handling**: Skips corrupted or un-processable PDFs without halting the entire process.
- **Environment-based Configuration**: Securely manages API keys using a `.env` file.
- **Sleep Prevention**: Built-in support for `caffeinate` to prevent system sleep during long runs.

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

### Basic Usage

```bash
python app.py --index_name <your-pinecone-index-name> --pdf_folder <path/to/your/pdf/folder>
```

### Resume Processing

The script includes robust resume functionality. If processing is interrupted, you can continue where you left off:

```bash
python app.py --index_name <your-pinecone-index-name> --pdf_folder <path/to/your/pdf/folder> --resume
```

### Prevent System Sleep

For long-running processes, prevent your system from sleeping:

```bash
caffeinate -i python app.py --index_name <your-pinecone-index-name> --pdf_folder <path/to/your/pdf/folder>
```

### Check Progress

View current processing state at any time:

```bash
python view_state.py
```

### Example

If your PDFs are in a folder named `documents` and your Pinecone index is `my-legal-kb`:

**First run:**
```bash
caffeinate -i python app.py --index_name my-legal-kb --pdf_folder ./documents
```

**Resume after interruption:**
```bash
caffeinate -i python app.py --index_name my-legal-kb --pdf_folder ./documents --resume
```

The script will start processing the files and display a progress bar.

```
📁 Found 194 total PDF files
✅ Already processed: 45 files
⏳ Remaining to process: 149 files
Processing PDFs: 100%|██████████| 149/149 [45:32<00:00, 18.3s/it]
✅ Finished processing 149 PDF files.
Chunks created this session: 19,234
Total chunks in index: 25,873
📊 State saved to: processing_state.json
```

## Configuration

The script uses several configurable parameters in the `app.py` file:

- `EMBEDDING_MODEL`: OpenAI embedding model (default: "text-embedding-3-large")
- `CHUNK_SIZE`: Target size of each chunk in tokens (default: 1000)
- `CHUNK_OVERLAP`: Number of tokens to overlap between chunks (default: 100)
- `MAX_METADATA_CONTENT_LENGTH`: Maximum characters in metadata content (default: 35000)

### Command Line Options

- `--index_name`: Name of the Pinecone index (required)
- `--pdf_folder`: Path to folder containing PDF files (required)
- `--resume`: Resume from previous processing state
- `--state_file`: Custom path for state file (default: "processing_state.json")

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

## Resume and State Management

The script automatically creates a `processing_state.json` file that tracks:
- Which files have been successfully processed
- Number of chunks created per file
- Processing timestamps
- Total chunk count

This allows you to:
- Resume processing after interruptions
- Track progress across multiple sessions
- Avoid reprocessing files that were already completed
- Debug processing issues by reviewing state history

Use `python view_state.py` to view current processing state at any time.

## Error Handling

The script includes robust error handling:
- Skips corrupted or unreadable PDF files
- Handles OpenAI API token limits with intelligent batching
- Manages Pinecone metadata size limits automatically
- Validates Pinecone index dimensions before processing
- Saves progress after each successful file (interruption-safe)
- Provides detailed error messages for troubleshooting

## License

This project is licensed under the terms specified in the LICENSE file.
