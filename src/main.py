import os
import argparse
from ingestion import ingest_all_r_files
from retrieval import RPackageRagSystem

def main():
    parser = argparse.ArgumentParser(description="R Package RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest R package files")
    ingest_parser.add_argument("--content-dir", required=True, help="Directory containing all R-related files (.md, .R, .Rmd, .qmd)")
    ingest_parser.add_argument("--output-dir", default="./chroma_db", help="Directory to store vector database")
    ingest_parser.add_argument("--collection-name", default="r_knowledge_base", help="Name for the vector database collection")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("--db-path", required=True, help="Path to vector database")
    query_parser.add_argument("--question", help="Question to ask (if not provided, enters interactive mode)")
    query_parser.add_argument("--api-key", help="Anthropic API key (if not set as env var)")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, args.collection_name), exist_ok=True)
        
        db_path = os.path.join(args.output_dir, args.collection_name)
        
        print(f"Ingesting all R-related files from {args.content_dir}...")
        db = ingest_all_r_files(
            directory_path=args.content_dir, 
            collection_name=args.collection_name,
            output_dir=args.output_dir
        )
        
        print(f"Ingestion complete. Vector database stored in: {db_path}")
        
    elif args.command == "query":
        # Get API key from args or environment
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Warning: No Anthropic API key provided. Set it with --api-key or ANTHROPIC_API_KEY environment variable.")
        
        # Initialize RAG system with the same DB for both docs and code (we combined them)
        rag = RPackageRagSystem(args.db_path, args.db_path, api_key)
        
        if args.question:
            # Single question mode
            answer = rag.query(args.question)
            print(answer)
        else:
            # Interactive mode
            rag.interactive_mode()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()