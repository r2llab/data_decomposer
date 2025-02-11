from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
from queue import Queue

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MemoryQueues:
    """Class to hold the global and local memory queues for the ReSP pipeline"""
    global_evidence: Queue  # Stores summaries relevant to main question Q
    local_pathway: Queue   # Stores summaries relevant to sub-question Q*

from ..reasoner.llm_reasoner import LLMReasoner
from ..retriever.vector_retriever import VectorRetriever
from ..summarizer.llm_summarizer import LLMSummarizer
from ..generator.llm_generator import LLMGenerator

class ReSPPipeline:
    def __init__(self, 
                 index_path: Optional[Path] = None, 
                 openai_api_key: Optional[str] = None,
                 max_iterations: int = 5):
        """
        Initialize the ReSP pipeline components.
        
        Args:
            index_path: Optional path to load an existing discovery index
            openai_api_key: Optional OpenAI API key
            max_iterations: Maximum number of iterations for the recursive process
        """
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for ReSP pipeline")
            
        # Initialize the four main components
        self.reasoner = LLMReasoner(api_key=openai_api_key)
        self.retriever = VectorRetriever(index_path=index_path, api_key=openai_api_key)
        self.summarizer = LLMSummarizer(api_key=openai_api_key)
        self.generator = LLMGenerator(api_key=openai_api_key)
        
        # Configuration
        self.max_iterations = max_iterations
        self.openai_api_key = openai_api_key
        
        # Initialize memory queues
        self.memory = MemoryQueues(
            global_evidence=Queue(),
            local_pathway=Queue()
        )

    def run_query(self, query: str) -> Dict[str, Any]:
        """
        Run the complete ReSP pipeline on a given query.
        
        Args:
            query: The natural language query string (Q)
            
        Returns:
            Dict containing the answer and metadata
        """
        logger.info(f"\n\n=== Processing Query: {query} ===")
        
        # Initialize for new query
        self._reset_memory_queues()
        current_iteration = 0
        current_sub_question = query  # Initial sub-question is the main question
        
        while current_iteration < self.max_iterations:
            logger.info(f"\n--- Iteration {current_iteration + 1} ---")
            # Log the current sub-question being processed
            logger.info(f"Processing sub-question: {current_sub_question}")
            # 1. Retrieval Phase
            retrieved_docs = self._retrieve(current_sub_question)
            if not retrieved_docs:
                logger.warning("No relevant documents found")
                break
                
            # Log retrieved documents
            logger.info("Retrieved documents:")
            for i, doc in enumerate(retrieved_docs, 1):
                logger.info(f"\nDocument {i}:")
                logger.info(f"Content: {doc.get('content', 'No content')}")
                logger.info(f"Metadata: {doc.get('metadata', {})}")

            # 2. Summarization Phase
            self._summarize(
                documents=retrieved_docs,
                main_question=query,
                sub_question=current_sub_question
            )
            
            # 3. Reasoning Phase (except for first iteration)
            # if current_iteration > 0:
            reasoning_result = self._reason(
                main_question=query,
                current_sub_question=current_sub_question
            )

            # Log the reasoning result
            logger.info(f"Reasoning result: {reasoning_result}")
            
            if reasoning_result.get("should_exit", False):
                logger.info("Reasoner decided to exit iteration")
                break
                
            current_sub_question = reasoning_result.get("next_sub_question")
            if not current_sub_question:
                logger.info("No further sub-questions generated")
                break
            
            current_iteration += 1
        
        # 4. Generation Phase
        final_answer = self._generate(main_question=query)
        
        return final_answer

    def _reset_memory_queues(self):
        """Reset both memory queues for a new query"""
        self.memory.global_evidence = Queue()
        self.memory.local_pathway = Queue()

    def _retrieve(self, sub_question: str) -> List[Dict[str, Any]]:
        """Use the retriever to find relevant documents"""
        if self.retriever is None:
            raise NotImplementedError("Retriever component not implemented")
        return self.retriever.retrieve(sub_question)

    def _summarize(self, documents: List[Dict[str, Any]], main_question: str, sub_question: str):
        """Use the summarizer to process documents and update memory queues"""
        if self.summarizer is None:
            raise NotImplementedError("Summarizer component not implemented")
        
        summaries = self.summarizer.summarize(
            documents=documents,
            main_question=main_question,
            sub_question=sub_question
        )

        # Log the summaries
        logger.info(f"Global summary: {summaries.get('global_summary')}")
        logger.info(f"Local summary: {summaries.get('local_summary')}")
        
        # Update memory queues
        self.memory.global_evidence.put(summaries.get("global_summary"))
        self.memory.local_pathway.put(summaries.get("local_summary"))

    def _reason(self, main_question: str, current_sub_question: str) -> Dict[str, Any]:
        """Use the reasoner to decide next steps"""
        if self.reasoner is None:
            raise NotImplementedError("Reasoner component not implemented")
            
        return self.reasoner.reason(
            main_question=main_question,
            current_sub_question=current_sub_question,
            global_evidence=list(self.memory.global_evidence.queue),
            local_pathway=list(self.memory.local_pathway.queue)
        )

    def _generate(self, main_question: str) -> Dict[str, Any]:
        """Use the generator to produce final answer"""
        if self.generator is None:
            raise NotImplementedError("Generator component not implemented")
            
        return self.generator.generate(
            question=main_question,
            global_evidence=list(self.memory.global_evidence.queue),
            local_pathway=list(self.memory.local_pathway.queue)
        )
