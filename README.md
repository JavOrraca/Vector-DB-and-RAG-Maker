# Vector DB and RAG Maker

A vector database and Retrieval-Augmented Generation (RAG) system for coding in R, designed to provide bleeding-edge responses based on your local documentation.

## Overview

This system:
1. Ingests R documentation (`.md`, `.R`, `.Rmd`, `.qmd`) from a single directory
2. Creates vector embeddings of the content and stores them in a `chromadb` vector database
3. Provides a query interface to ask questions about the R packages
4. Uses an LLM to generate responses based on retrieved context

## Requirements

- Python 3.8+
- Required Python packages:
  - langchain
  - langchain-community
  - langchain-anthropic (for Claude 3.7 Sonnet integration)
  - langchain-text-splitters
  - chromadb
  - sentence-transformers
  - argparse
  - glob

## Installation

1. Clone this repository:
```bash
git clone JavOrraca/Vector-DB-and-RAG-Maker
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY=your-api-key
```

## Usage

### Ingesting R-related Files

First, ingest all your R-related files from a single directory:

```bash
python src/main.py ingest --content-dir ./data --output-dir ./chroma_db
```

This command will:
- Find all `.md`, `.R`, `.Rmd`, and `.qmd` files in the specified directory
- Process them appropriately based on file type
- Store them in a single unified Chroma vector database

### Querying the System

After ingestion, you can query the system:

```bash
# Interactive mode
python src/main.py query --db-path ./chroma_db/r_knowledge_base

# Single query mode
python src/main.py query --db-path ./chroma_db/r_knowledge_base --question "How do I use dplyr's filter function?"
```

## Supported File Types

The system can ingest and process the following file types:

- **Markdown (`.md`)** - Documentation, READMEs, etc.
- **R Files (`.R`)** - R source code files
- **R Markdown (`.Rmd`)** - Mixed R code and markdown
- **Quarto (`.qmd`)** - Next-gen technical publishing framework (basically the successor to `.Rmd`)

All files are processed appropriately based on their type and structure. If you want any additional file types, please reach out to Javier.

## System Components

### Ingestion

The ingestion pipeline:
1. Recursively finds all supported files in the specified directory
2. Processes each file type appropriately:
   - Splits markdown files by headers and then into chunks
   - Splits R code files into chunks
   - Handles `.Rmd` and `.qmd` files intelligently, attempting to parse them as markdown first
3. Creates vector embeddings for each chunk
4. Stores the embeddings in a unified Chroma vector database

### Retrieval

The retrieval system:
1. Takes a user question
2. Searches the vector database for relevant context
3. Combines the results
4. Sends the most relevant context to an LLM
5. Returns the LLM's response

## Customization

### Embedding Models

By default, the system uses the `sentence-transformers/all-MiniLM-L6-v2` model for embeddings. You can modify this in the code to use other models.

### LLM

The system is configured to use Anthropic's Claude 3.7 Sonnet, but you can modify it to use other LLMs supported by LangChain.

### Chunking Parameters

You can adjust the chunking parameters in the code to better suit your needs:
- `chunk_size`: The size of text chunks
- `chunk_overlap`: The amount of overlap between chunks
