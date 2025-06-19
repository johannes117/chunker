import os
import argparse
import json
from datetime import datetime
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

# Pinecone metadata configuration
MAX_METADATA_CONTENT_LENGTH = 35000  # Conservative limit to stay under 40KB metadata limit
PINECONE_BATCH_SIZE = 50  # Conservative batch size for upserts to avoid 4MB message limit

# --- Argument Parsing ---
def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Process PDFs and upload to Pinecone.")
    parser.add_argument("--index_name", type=str, required=True, help="Name of the Pinecone index.")
    parser.add_argument("--pdf_folder", type=str, required=True, help="Path to the folder containing PDF files.")
    parser.add_argument("--resume", action="store_true", help="Resume from previous state file.")
    parser.add_argument("--state_file", type=str, default="processing_state.json", help="Path to state file for resume functionality.")
    return parser.parse_args()

# --- State Management Functions ---
def load_state(state_file: str) -> dict:
    """Load processing state from JSON file."""
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load state file {state_file}: {e}")
    return {"processed_files": [], "total_chunks": 0, "started_at": None}

def save_state(state_file: str, state: dict):
    """Save processing state to JSON file."""
    try:
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"⚠️  Warning: Could not save state file {state_file}: {e}")

def update_state(state: dict, filename: str, chunks_added: int):
    """Update state with processed file."""
    state["processed_files"].append({
        "filename": filename,
        "processed_at": datetime.now().isoformat(),
        "chunks_added": chunks_added
    })
    state["total_chunks"] += chunks_added
    return state

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

def get_embeddings(texts: list[str], client: OpenAI, tokenizer, model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """Generates embeddings for a list of text chunks, processing in batches to avoid token limits."""
    texts = [t.replace("\n", " ") for t in texts]
    all_embeddings = []
    
    # Process in smaller batches to stay under the 300k token limit
    current_batch = []
    current_token_count = 0
    max_tokens_per_batch = 180000  # Even more conservative limit
    batch_count = 0
    
    for i, text in enumerate(texts):
        text_tokens = len(tokenizer.encode(text))
        
        # If adding this text would exceed the limit, process current batch
        if current_token_count + text_tokens > max_tokens_per_batch and current_batch:
            batch_count += 1
            # Double-check actual token count before sending
            actual_tokens = sum(len(tokenizer.encode(t)) for t in current_batch)
            print(f"    📦 Processing batch {batch_count} with {len(current_batch)} chunks (counted: {current_token_count:,}, actual: {actual_tokens:,} tokens)")
            
            if actual_tokens > 300000:
                print(f"  └── ❌ Batch too large! Actual tokens ({actual_tokens:,}) exceed 300k limit")
                return []
            
            try:
                response = client.embeddings.create(input=current_batch, model=model)
                all_embeddings.extend([item.embedding for item in response.data])
            except Exception as e:
                print(f"  └── ❌ Error generating embeddings for batch {batch_count}: {e}")
                print(f"      Batch had {len(current_batch)} chunks with actual {actual_tokens:,} tokens")
                return []
            
            # Start new batch
            current_batch = [text]
            current_token_count = text_tokens
        else:
            current_batch.append(text)
            current_token_count += text_tokens
    
    # Process the final batch
    if current_batch:
        batch_count += 1
        actual_tokens = sum(len(tokenizer.encode(t)) for t in current_batch)
        print(f"    📦 Processing final batch {batch_count} with {len(current_batch)} chunks (counted: {current_token_count:,}, actual: {actual_tokens:,} tokens)")
        
        if actual_tokens > 300000:
            print(f"  └── ❌ Final batch too large! Actual tokens ({actual_tokens:,}) exceed 300k limit")
            return []
        
        try:
            response = client.embeddings.create(input=current_batch, model=model)
            all_embeddings.extend([item.embedding for item in response.data])
        except Exception as e:
            print(f"  └── ❌ Error generating embeddings for final batch {batch_count}: {e}")
            print(f"      Batch had {len(current_batch)} chunks with actual {actual_tokens:,} tokens")
            return []
    
    return all_embeddings

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

    # --- State Management ---
    state = load_state(args.state_file)
    if not state["started_at"]:
        state["started_at"] = datetime.now().isoformat()
    
    processed_filenames = [item["filename"] for item in state["processed_files"]]
    
    # --- File Processing ---
    pdf_files = [f for f in os.listdir(args.pdf_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in '{args.pdf_folder}'.")
        return
    
    # Filter out already processed files if resuming
    if args.resume and processed_filenames:
        remaining_files = [f for f in pdf_files if f not in processed_filenames]
        print(f"📁 Found {len(pdf_files)} total PDF files")
        print(f"✅ Already processed: {len(processed_filenames)} files")
        print(f"⏳ Remaining to process: {len(remaining_files)} files")
        pdf_files = remaining_files
    else:
        print(f"Found {len(pdf_files)} PDF files to process. Starting fresh...")
    
    if not pdf_files:
        print("🎉 All files already processed!")
        return
        
    total_chunks_created = state["total_chunks"]

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
            total_tokens = sum(len(tokenizer.encode(chunk)) for chunk in text_chunks)
            print(f"  📄 {len(text_chunks)} chunks, {total_tokens:,} total tokens")
            embeddings = get_embeddings(text_chunks, openai_client, tokenizer)
            if not embeddings:
                pbar.update(1)
                continue
            
            # 4. Prepare vectors for Pinecone
            vectors_to_upsert = []
            for i, chunk in enumerate(text_chunks):
                if i < len(embeddings):
                    vector_id = str(uuid.uuid4())
                    # Truncate content to stay within Pinecone's 40KB metadata limit
                    # Reserve space for other metadata fields and JSON overhead
                    truncated_content = chunk[:MAX_METADATA_CONTENT_LENGTH]
                    if len(chunk) > MAX_METADATA_CONTENT_LENGTH:
                        truncated_content += "..."
                    
                    metadata = {
                        'content': truncated_content, 
                        'source': filename,
                        'chunk_index': i,
                        'original_length': len(chunk)
                    }
                    vectors_to_upsert.append({
                        "id": vector_id,
                        "values": embeddings[i],
                        "metadata": metadata
                    })
            
            # 5. Upsert to Pinecone in batches
            chunks_added = 0
            if vectors_to_upsert:
                try:
                    # Use smaller batches to avoid 4MB message size limit
                    # Large documents can have huge payloads, so be conservative
                    batch_size = PINECONE_BATCH_SIZE
                    print(f"  📤 Upserting {len(vectors_to_upsert)} vectors in batches of {batch_size}")
                    
                    for j in range(0, len(vectors_to_upsert), batch_size):
                        batch = vectors_to_upsert[j:j + batch_size]
                        index.upsert(vectors=batch)
                        print(f"    ✅ Upserted batch {j//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1}")
                    
                    chunks_added = len(vectors_to_upsert)
                    total_chunks_created += chunks_added
                    
                    # Update and save state after successful processing
                    state = update_state(state, filename, chunks_added)
                    save_state(args.state_file, state)
                    
                except Exception as e:
                    print(f"  └── ❌ Error upserting to Pinecone for {filename}: {e}")
                    print(f"      Attempted to upsert {len(vectors_to_upsert)} vectors")
            
            pbar.update(1)

    # Calculate session stats
    initial_chunks = state["total_chunks"] - sum(item["chunks_added"] for item in state["processed_files"])
    session_chunks = total_chunks_created - initial_chunks
    
    print(f"\n✅ Finished processing {len(pdf_files)} PDF files.")
    print(f"Chunks created this session: {session_chunks}")
    print(f"Total chunks in index: {total_chunks_created}")
    print(f"📊 State saved to: {args.state_file}")

if __name__ == "__main__":
    main() 