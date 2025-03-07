from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

class RPackageRagSystem:
    """
    RAG system for querying R package documentation and code.
    """
    
    def __init__(self, docs_db_path, code_db_path, anthropic_api_key=None):
        """
        Initialize the RAG system with paths to the vector databases.
        
        Args:
            docs_db_path: Path to the Chroma DB for markdown documentation
            code_db_path: Path to the Chroma DB for R code
            anthropic_api_key: Optional Anthropic API key
        """
        # Load embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load vector databases
        self.docs_db = Chroma(
            persist_directory=docs_db_path,
            embedding_function=self.embedding_model
        )
        
        self.code_db = Chroma(
            persist_directory=code_db_path,
            embedding_function=self.embedding_model
        )
        
        # Initialize LLM (Claude 3.7 Sonnet)
        self.llm = ChatAnthropic(
            temperature=0,
            model="claude-3-7-sonnet-20250219",
            anthropic_api_key=anthropic_api_key
        )
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are an expert R programmer and data scientist. Use the provided context about R packages 
            to answer the user's question. The context includes both documentation and code from various R packages.
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer:
            """
        )
    
    def query(self, question, k=5, doc_weight=0.7):
        """
        Query the RAG system with a question.
        
        Args:
            question: User's question about R packages
            k: Number of documents to retrieve from each database
            doc_weight: Weight to give documentation vs code (0-1)
            
        Returns:
            Answer from the LLM
        """
        # Retrieve relevant documentation
        docs_results = self.docs_db.similarity_search_with_score(question, k=k)
        
        # Retrieve relevant code
        code_results = self.code_db.similarity_search_with_score(question, k=k)
        
        # Combine results with weighting
        combined_results = []
        
        # Add documentation with its weight
        for doc, score in docs_results:
            combined_results.append((doc, score * doc_weight))
        
        # Add code with its weight
        for doc, score in code_results:
            combined_results.append((doc, score * (1 - doc_weight)))
        
        # Sort by weighted score and take top k*2
        combined_results.sort(key=lambda x: x[1], reverse=True)
        top_results = combined_results[:k*2]
        
        # Extract documents
        context_docs = [item[0] for item in top_results]
        
        # Build context string
        context_str = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')} | Type: {doc.metadata.get('file_type', 'Unknown')}]\n{doc.page_content}"
            for doc in context_docs
        ])
        
        # Create chain
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.docs_db.as_retriever(search_kwargs={"k": k}),
            chain_type_kwargs={"prompt": self.prompt_template}
        )
        
        # Run chain
        result = chain.invoke({
            "query": question,
            "context": context_str
        })
        
        return result["result"]
    
    def interactive_mode(self):
        """
        Start an interactive session for querying the RAG system.
        """
        print("R Package RAG System - Interactive Mode (powered by Claude 3.7 Sonnet)")
        print("Type 'exit' to quit")
        
        while True:
            question = input("\nEnter your question: ")
            
            if question.lower() == 'exit':
                break
                
            try:
                answer = self.query(question)
                print("\nAnswer:")
                print(answer)
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    # Example usage
    docs_db_path = "./chroma_db_r_packages_docs"
    code_db_path = "./chroma_db_r_packages_code"
    
    # Initialize RAG system
    rag = RPackageRagSystem(docs_db_path, code_db_path)
    
    # Start interactive mode
    rag.interactive_mode()