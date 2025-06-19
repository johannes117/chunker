import os
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import pypdf
import tiktoken
from tqdm import tqdm
import uuid

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# OpenAI and Pinecone API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_CTX_LENGTH = 8191 # Max tokens for the model
EMBEDDING_DIMENSION = 3072 # Dimensions for text-embedding-3-large

# Text chunking configuration (in tokens)
CHUNK_SIZE = 1000  # Target size of each chunk in tokens
CHUNK_OVERLAP = 100 # How many tokens to overlap between chunks

# --- Argument Parsing ---
def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Process PDFs and upload to Pinecone.")
    parser.add_argument("--index_name", type=str, required=True, help="Name of the Pinecone index.")
    parser.add_argument("--pdf_folder", type=str, required=True, help="Path to the folder containing PDF files.")
    return parser.parse_args()

# --- Core Functions ---
def get_pdf_text(file_path: str) -> str:
    """Extracts text from a single PDF file."""
    try:
        reader = pypdf.PdfReader(file_path)
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        return text.replace("\t", " ").replace("  ", " ")
    except Exception as e:
        print(f"  └── ❌ Error reading {os.path.basename(file_path)}: {e}")
        return ""

def chunk_text_with_tokens(text: str, tokenizer) -> list[str]:
    """
    Chunks text into smaller pieces based on token count, respecting sentence boundaries.
    """
    if not text:
        return []

    # Split the text into sentences
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        sentence_length = len(tokenizer.encode(sentence))
        if current_length + sentence_length <= CHUNK_SIZE:
            current_chunk += sentence + '. '
            current_length += sentence_length
        else:
            # If a single sentence is too long, we must split it.
            if sentence_length > CHUNK_SIZE:
                # Force split the long sentence
                sub_sentences = [sentence[i:i+CHUNK_SIZE] for i in range(0, len(sentence), CHUNK_SIZE)]
                chunks.extend(sub_sentences)
            else:
                 # Add the completed chunk and start a new one
                chunks.append(current_chunk.strip())
                # Overlap the last few sentences
                overlap_text = ' '.join(current_chunk.split('. ')[-5:])
                current_chunk = overlap_text + '. ' + sentence + '. '
                current_length = len(tokenizer.encode(current_chunk))

    if current_chunk:
        chunks.append(current_chunk.strip())
        
    # Final check to ensure no chunk exceeds the model's absolute max length
    final_chunks = []
    for chunk in chunks:
        if len(tokenizer.encode(chunk)) > EMBEDDING_CTX_LENGTH:
            # If a chunk is still too long, truncate it. This is a fallback.
            truncated_tokens = tokenizer.encode(chunk)[:EMBEDDING_CTX_LENGTH]
            final_chunks.append(tokenizer.decode(truncated_tokens))
        else:
            final_chunks.append(chunk)

    return final_chunks

def get_embeddings(texts: list[str], client: OpenAI, model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """Generates embeddings for a list of text chunks."""
    texts = [t.replace("\n", " ") for t in texts]
    try:
        response = client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"  └── ❌ Error generating embeddings: {e}")
        return []

def main():
    """Main function to run the indexing pipeline."""
    args = parse_arguments()

    # --- Initialization ---
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        raise ValueError("API keys for OpenAI and Pinecone must be set in the .env file.")

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if the Pinecone index exists and has the correct dimension
    try:
        index_description = pinecone_client.describe_index(args.index_name)
        if index_description.dimension != EMBEDDING_DIMENSION:
            print(f"❌ Error: Index '{args.index_name}' has dimension {index_description.dimension}, but model '{EMBEDDING_MODEL}' requires {EMBEDDING_DIMENSION}.")
            return
    except Exception as e:
        print(f"❌ Error connecting to or describing index '{args.index_name}': {e}")
        print("Please ensure the index exists and your API key is correct.")
        return
        
    index = pinecone_client.Index(args.index_name)
    tokenizer = tiktoken.get_encoding("cl100k_base") # Tokenizer for text-embedding models

    # --- File Processing ---
    pdf_files = [f for f in os.listdir(args.pdf_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in '{args.pdf_folder}'.")
        return
        
    total_chunks_created = 0
    print(f"Found {len(pdf_files)} PDF files to process. Starting ingestion...")

    with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
        for filename in pdf_files:
            pbar.set_postfix_str(filename, refresh=True)
            
            # 1. Extract text
            file_path = os.path.join(args.pdf_folder, filename)
            document_text = get_pdf_text(file_path)
            if not document_text:
                pbar.update(1)
                continue

            # 2. Chunk text
            text_chunks = chunk_text_with_tokens(document_text, tokenizer)
            if not text_chunks:
                pbar.update(1)
                continue

            # 3. Generate embeddings
            embeddings = get_embeddings(text_chunks, openai_client)
            if not embeddings:
                pbar.update(1)
                continue
            
            # 4. Prepare vectors for Pinecone
            vectors_to_upsert = []
            for i, chunk in enumerate(text_chunks):
                if i < len(embeddings):
                    vector_id = str(uuid.uuid4())
                    metadata = {'content': chunk, 'source': filename}
                    vectors_to_upsert.append({
                        "id": vector_id,
                        "values": embeddings[i],
                        "metadata": metadata
                    })
            
            # 5. Upsert to Pinecone in batches
            if vectors_to_upsert:
                try:
                    # Upsert in smaller batches if necessary
                    batch_size = 100 
                    for j in range(0, len(vectors_to_upsert), batch_size):
                        batch = vectors_to_upsert[j:j + batch_size]
                        index.upsert(vectors=batch)
                    total_chunks_created += len(vectors_to_upsert)
                except Exception as e:
                    print(f"  └── ❌ Error upserting to Pinecone for {filename}: {e}")
            
            pbar.update(1)

    print(f"\n✅ Finished processing all {len(pdf_files)} PDF files.")
    print(f"Total chunks created: {total_chunks_created}")

if __name__ == "__main__":
    main() 