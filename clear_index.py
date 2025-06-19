#!/usr/bin/env python3
"""
Simple script to clear all vectors from a Pinecone index.
"""
import os
import argparse
from dotenv import load_dotenv
from pinecone import Pinecone

def main():
    parser = argparse.ArgumentParser(description="Clear all vectors from a Pinecone index.")
    parser.add_argument("--index_name", type=str, required=True, help="Name of the Pinecone index to clear.")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    
    if not api_key:
        print("❌ Error: PINECONE_API_KEY not found in .env file")
        return
    
    try:
        # Connect to Pinecone
        pc = Pinecone(api_key=api_key)
        index = pc.Index(args.index_name)
        
        # Get current stats
        stats = index.describe_index_stats()
        vector_count = stats.total_vector_count
        
        print(f"📊 Current index '{args.index_name}' has {vector_count} vectors")
        
        if vector_count == 0:
            print("✅ Index is already empty!")
            return
        
        # Confirm deletion
        response = input(f"❓ Are you sure you want to delete all {vector_count} vectors? (yes/no): ")
        
        if response.lower() in ['yes', 'y']:
            print("🗑️  Deleting all vectors...")
            index.delete(delete_all=True)
            print("✅ Index cleared successfully!")
        else:
            print("❌ Operation cancelled.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 