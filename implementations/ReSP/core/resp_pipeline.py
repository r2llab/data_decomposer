from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import os
from dataclasses import dataclass
from queue import Queue
import re

# Configure logging
def setup_logging(log_file=None):
    """Configure logging to output to both console and file if specified"""
    handlers = [
        logging.StreamHandler()  # Console handler
    ]
    
    # Add file handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

# Default logger (console only)
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
from ..utils.cost_tracker import CostTracker

class ReSPPipeline:
    def __init__(self, 
                 index_path: Optional[Path] = None, 
                 openai_api_key: Optional[str] = None,
                 max_iterations: int = 5,
                 log_file: Optional[str] = None,
                 cost_tracker: Optional[CostTracker] = None):
        """
        Initialize the ReSP pipeline components.
        
        Args:
            index_path: Optional path to load an existing discovery index
            openai_api_key: Optional OpenAI API key
            max_iterations: Maximum number of iterations for the recursive process
            log_file: Optional path to a log file where logs will be written
            cost_tracker: Optional cost tracker instance to track API usage
        """
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for ReSP pipeline")
        
        # Setup logging
        self.logger = setup_logging(log_file)
        self.logger.info("Initializing ReSP Pipeline")
        
        # Initialize cost tracker or use provided one
        self.cost_tracker = cost_tracker or CostTracker()
            
        # Initialize the four main components
        self.reasoner = LLMReasoner(api_key=openai_api_key, cost_tracker=self.cost_tracker)
        self.retriever = VectorRetriever(index_path=index_path, api_key=openai_api_key)
        self.summarizer = LLMSummarizer(api_key=openai_api_key, cost_tracker=self.cost_tracker)
        self.generator = LLMGenerator(api_key=openai_api_key, cost_tracker=self.cost_tracker)

        # Configuration
        self.max_iterations = max_iterations
        self.openai_api_key = openai_api_key
        self.log_file = log_file
        
        # Initialize memory queues
        self.memory = MemoryQueues(
            global_evidence=Queue(),
            local_pathway=Queue()
        )
        
        # Track document sources
        self.document_sources = set()

    def run_query(self, query: str) -> Dict[str, Any]:
        """
        Run the complete ReSP pipeline on a given query.
        
        Args:
            query: The natural language query string (Q)
            
        Returns:
            Dict containing the answer and metadata
        """
        self.logger.info(f"\n\n=== Processing Query: {query} ===")
        
        # Initialize for new query
        self._reset_memory_queues()
        self.document_sources = set()  # Reset document sources
        
        # Reset cost tracking for this query
        self.cost_tracker.reset_query_stats()
        
        current_iteration = 0
        current_sub_question = query  # Initial sub-question is the main question
        
        while current_iteration < self.max_iterations:
            self.logger.info(f"\n--- Iteration {current_iteration + 1} ---")
            # Log the current sub-question being processed
            self.logger.info(f"Processing sub-question: {current_sub_question}")
            # 1. Retrieval Phase
            retrieved_docs = self._retrieve(current_sub_question)
            if not retrieved_docs:
                self.logger.warning("No relevant documents found")
                break
                
            # Log retrieved documents
            self.logger.info("Retrieved documents:")
            for i, doc in enumerate(retrieved_docs, 1):
                self.logger.info(f"\nDocument {i}:")
                self.logger.info(f"Content: {doc.get('content', 'No content')}")
                self.logger.info(f"Metadata: {doc.get('metadata', {})}")
                
                # Track document sources
                if 'metadata' in doc and 'source' in doc['metadata']:
                    source = doc['metadata']['source']
                    if source and source != "unknown":
                        self.document_sources.add(source)

            # 2. Summarization Phase
            self._summarize(
                documents=retrieved_docs,
                main_question=query,
                sub_question=current_sub_question
            )
            
            # 3. Reasoning Phase
            next_step = self._reason(
                main_question=query,
                current_sub_question=current_sub_question
            )
            
            # Log reasoning output
            self.logger.info(f"Reasoning result: {next_step}")
            
            # Decide what to do next
            if not next_step.get("should_exit", False):
                # If we should continue, update the sub-question
                current_sub_question = next_step.get("next_sub_question")
                if not current_sub_question:
                    # If no new sub-question is provided, exit the loop
                    self.logger.info("No further sub-questions generated, generating final answer")
                    break
                current_iteration += 1
                self.logger.info(f"Continuing with sub-question: {current_sub_question}")
            else:
                # If we're done, generate the final answer
                self.logger.info("Reasoning complete, generating final answer")
                break
        
        # Log document sources
        self.logger.info(f"Document sources used: {self.document_sources}")
                
        # Generate final answer
        final_result = self._generate(
            main_question=query
        )
        
        # Ensure document sources are in the result
        if 'document_sources' not in final_result:
            final_result['document_sources'] = list(self.document_sources)
        
        # Add cost metrics to the result
        cost_summary = self.cost_tracker.get_query_summary()
        final_result['cost_metrics'] = {
            'total_cost': float(cost_summary['query_cost']),
            'total_tokens': int(cost_summary['query_tokens']),
            'api_calls': int(cost_summary['query_calls']),
            'model_breakdown': {
                model: {
                    'cost': float(stats['cost']),
                    'tokens': int(stats['tokens']),
                    'calls': int(stats['calls'])
                }
                for model, stats in cost_summary['models'].items()
            },
            'endpoint_breakdown': {
                endpoint: {
                    'cost': float(stats['cost']),
                    'tokens': int(stats['tokens']),
                    'calls': int(stats['calls'])
                }
                for endpoint, stats in cost_summary['endpoints'].items()
            }
        }
        
        # Log cost information
        self.logger.info(f"Query cost: ${cost_summary['query_cost']:.6f}")
        self.logger.info(f"Total tokens: {cost_summary['query_tokens']}")
        self.logger.info(f"API calls: {cost_summary['query_calls']}")
        
        self.logger.info(f"Final answer: {final_result.get('answer')}")
        
        return final_result

    def _reset_memory_queues(self):
        """Reset memory queues for a new query"""
        # Clear existing queues
        self.memory.global_evidence = Queue()
        self.memory.local_pathway = Queue()
        
        # Reset document sources
        self.document_sources = set()

    def _retrieve(self, question: str) -> List[Dict[str, Any]]:
        """Use the retriever to get relevant documents"""
        if self.retriever is None:
            raise NotImplementedError("Retriever component not implemented")
            
        retrieved_docs = self.retriever.retrieve(question)
        
        # Capture document sources from metadata
        for doc in retrieved_docs:
            if 'metadata' in doc and 'source' in doc['metadata']:
                source = doc['metadata']['source']
                if source and source != "unknown":
                    self.document_sources.add(source)
            
        # Log tracked sources
        self.logger.info(f"Currently tracked document sources: {self.document_sources}")
            
        return retrieved_docs

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
        self.logger.info(f"Global summary: {summaries.get('global_summary')}")
        self.logger.info(f"Local summary: {summaries.get('local_summary')}")
        
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
        
        # Ensure document_sources is a list, not a set
        document_sources = list(self.document_sources) if self.document_sources else []
        
        # Log document sources being passed to generator
        self.logger.info(f"Passing {len(document_sources)} document sources to generator: {document_sources}")
        
        # Call generator with document sources
        result = self.generator.generate(
            question=main_question,
            global_evidence=list(self.memory.global_evidence.queue),
            local_pathway=list(self.memory.local_pathway.queue),
            document_sources=document_sources
        )
        
        # Ensure document sources are in the result
        if not result.get('document_sources') and document_sources:
            result['document_sources'] = document_sources
            
        return result
