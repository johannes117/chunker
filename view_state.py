#!/usr/bin/env python3
"""
Simple script to view the current processing state.
"""
import json
import os
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="View processing state.")
    parser.add_argument("--state_file", type=str, default="processing_state.json", help="Path to state file.")
    args = parser.parse_args()
    
    if not os.path.exists(args.state_file):
        print(f"❌ State file '{args.state_file}' not found.")
        return
    
    try:
        with open(args.state_file, 'r') as f:
            state = json.load(f)
        
        print(f"📊 Processing State Summary")
        print(f"=" * 50)
        
        if state.get("started_at"):
            started = datetime.fromisoformat(state["started_at"])
            print(f"Started: {started.strftime('%Y-%m-%d %H:%M:%S')}")
        
        processed_files = state.get("processed_files", [])
        total_chunks = state.get("total_chunks", 0)
        
        print(f"Files processed: {len(processed_files)}")
        print(f"Total chunks: {total_chunks}")
        
        if processed_files:
            print(f"\n📋 Recently Processed Files:")
            # Show last 10 files
            for item in processed_files[-10:]:
                processed_time = datetime.fromisoformat(item["processed_at"])
                print(f"  • {item['filename']} ({item['chunks_added']} chunks) - {processed_time.strftime('%H:%M:%S')}")
            
            if len(processed_files) > 10:
                print(f"  ... and {len(processed_files) - 10} more files")
        
    except Exception as e:
        print(f"❌ Error reading state file: {e}")

if __name__ == "__main__":
    main() 