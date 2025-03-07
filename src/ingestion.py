import os
import glob
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def ingest_all_r_files(directory_path, collection_name="r_knowledge_base", output_dir="./chroma_db"):
    """
    Ingest all R-related files (R, Rmd, qmd, md) from a single directory, 
    split into appropriate chunks, and store in vector database.
    
    Args:
        directory_path: Path to directory containing files to ingest
        collection_name: Name of the collection in the vector database
        output_dir: Base directory to store the vector database
    """
    # Find all relevant files by type
    markdown_files = glob.glob(os.path.join(directory_path, "**/*.md"), recursive=True)
    r_files = glob.glob(os.path.join(directory_path, "**/*.R"), recursive=True)
    rmd_files = glob.glob(os.path.join(directory_path, "**/*.Rmd"), recursive=True)
    qmd_files = glob.glob(os.path.join(directory_path, "**/*.qmd"), recursive=True)
    
    # Headers to split markdown on
    headers_to_split_on = [
        ("#", "header1"),
        ("##", "header2"),
        ("###", "header3"),
        ("####", "header4")
    ]
    
    # Initialize splitters
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    code_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Process all files
    documents = []
    
    # Process markdown files
    print(f"Processing {len(markdown_files)} markdown (.md) files...")
    for file_path in markdown_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                content = f.read()
            except UnicodeDecodeError:
                print(f"Warning: Could not read {file_path} due to encoding issues. Skipping.")
                continue
        
        # Get relative path for metadata
        rel_path = os.path.relpath(file_path, directory_path)
        
        # Split by headers first
        md_docs = markdown_splitter.split_text(content)
        for doc in md_docs:
            doc.metadata["source"] = rel_path
            doc.metadata["file_type"] = "markdown"
        
        # Further split by size if needed
        docs = text_splitter.split_documents(md_docs)
        documents.extend(docs)
    
    # Process R files
    print(f"Processing {len(r_files)} R (.R) files...")
    for file_path in r_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                content = f.read()
            except UnicodeDecodeError:
                print(f"Warning: Could not read {file_path} due to encoding issues. Skipping.")
                continue
        
        # Get relative path for metadata
        rel_path = os.path.relpath(file_path, directory_path)
        
        # Split text into chunks
        chunks = code_splitter.create_documents(
            texts=[content],
            metadatas=[{"source": rel_path, "file_type": "R", "language": "R"}]
        )
        documents.extend(chunks)
    
    # Process Rmd files
    print(f"Processing {len(rmd_files)} R Markdown (.Rmd) files...")
    for file_path in rmd_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                content = f.read()
            except UnicodeDecodeError:
                print(f"Warning: Could not read {file_path} due to encoding issues. Skipping.")
                continue
        
        # Get relative path for metadata
        rel_path = os.path.relpath(file_path, directory_path)
        
        # Try to split by headers first (since Rmd is markdown-based)
        try:
            md_docs = markdown_splitter.split_text(content)
            for doc in md_docs:
                doc.metadata["source"] = rel_path
                doc.metadata["file_type"] = "Rmd"
            
            # Further split by size if needed
            docs = text_splitter.split_documents(md_docs)
            documents.extend(docs)
        except Exception as e:
            # Fallback to regular splitting if header parsing fails
            print(f"Warning: Markdown parsing failed for {file_path}, using regular chunking")
            chunks = text_splitter.create_documents(
                texts=[content],
                metadatas=[{"source": rel_path, "file_type": "Rmd"}]
            )
            documents.extend(chunks)
    
    # Process Quarto files
    print(f"Processing {len(qmd_files)} Quarto (.qmd) files...")
    for file_path in qmd_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                content = f.read()
            except UnicodeDecodeError:
                print(f"Warning: Could not read {file_path} due to encoding issues. Skipping.")
                continue
        
        # Get relative path for metadata
        rel_path = os.path.relpath(file_path, directory_path)
        
        # Try to split by headers first (since qmd is markdown-based)
        try:
            md_docs = markdown_splitter.split_text(content)
            for doc in md_docs:
                doc.metadata["source"] = rel_path
                doc.metadata["file_type"] = "qmd"
            
            # Further split by size if needed
            docs = text_splitter.split_documents(md_docs)
            documents.extend(docs)
        except Exception as e:
            # Fallback to regular splitting if header parsing fails
            print(f"Warning: Markdown parsing failed for {file_path}, using regular chunking")
            chunks = text_splitter.create_documents(
                texts=[content],
                metadatas=[{"source": rel_path, "file_type": "qmd"}]
            )
            documents.extend(chunks)
    
    # Store in vector database
    print(f"Creating vector database with {len(documents)} document chunks...")
    # Use os.path.join for proper path handling
    persist_dir = os.path.join(output_dir, collection_name)
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    
    print(f"Vector database created and persisted to {persist_dir}")
    return db

# For backward compatibility
def ingest_markdown_files(directory_path, collection_name="r_packages_docs", output_dir="./chroma_db"):
    """
    Legacy function. Use ingest_all_r_files instead.
    """
    print("Warning: This function is deprecated. Use ingest_all_r_files instead.")
    
    # Find all .md files in the directory
    md_files = glob.glob(os.path.join(directory_path, "**/*.md"), recursive=True)
    
    # Headers to split markdown on
    headers_to_split_on = [
        ("#", "header1"),
        ("##", "header2"),
        ("###", "header3"),
        ("####", "header4")
    ]
    
    # Initialize splitters
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Process each file
    documents = []
    for file_path in md_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                content = f.read()
            except UnicodeDecodeError:
                print(f"Warning: Could not read {file_path} due to encoding issues. Skipping.")
                continue
        
        # Get relative path for metadata
        rel_path = os.path.relpath(file_path, directory_path)
        
        # Split by headers first
        md_docs = markdown_splitter.split_text(content)
        for doc in md_docs:
            doc.metadata["source"] = rel_path
            doc.metadata["file_type"] = "markdown"
        
        # Further split by size if needed
        docs = text_splitter.split_documents(md_docs)
        documents.extend(docs)
    
    # Store in vector database
    # Use os.path.join for proper path handling
    persist_dir = os.path.join(output_dir, collection_name)
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    
    return db

# For backward compatibility
def ingest_r_files(directory_path, collection_name="r_packages_code", output_dir="./chroma_db"):
    """
    Legacy function. Use ingest_all_r_files instead.
    """
    print("Warning: This function is deprecated. Use ingest_all_r_files instead.")
    
    # Find all .R files in the directory
    r_files = glob.glob(os.path.join(directory_path, "**/*.R"), recursive=True)
    
    # Initialize splitter for code
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Process each file
    documents = []
    for file_path in r_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                content = f.read()
            except UnicodeDecodeError:
                print(f"Warning: Could not read {file_path} due to encoding issues. Skipping.")
                continue
        
        # Get relative path for metadata
        rel_path = os.path.relpath(file_path, directory_path)
        
        # Split text into chunks
        chunks = text_splitter.create_documents(
            texts=[content],
            metadatas=[{"source": rel_path, "file_type": "R", "language": "R"}]
        )
        documents.extend(chunks)
    
    # Store in vector database
    # Use os.path.join for proper path handling
    persist_dir = os.path.join(output_dir, collection_name)
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    
    return db

if __name__ == "__main__":
    # Example usage
    # Replace with your actual path
    r_content_path = "../data/r_content"
    output_dir = "./chroma_db"
    
    # Use the new unified function
    db = ingest_all_r_files(r_content_path, output_dir=output_dir)
    
    print(f"Ingested all R-related files into Chroma DB")