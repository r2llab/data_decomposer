import argparse
import logging
from core.config import ConfigurationManager
from core.factory import ImplementationFactory
# Import implementations to register them
import implementations

def setup_logging(config):
    """Setup logging based on configuration."""
    logging.basicConfig(
        level=getattr(logging, config.get('logging', {}).get('level', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=config.get('logging', {}).get('file')
    )

def main():
    print("Starting main")
    parser = argparse.ArgumentParser(description='R2L Query Processing System')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('query', type=str, help='Query to process')
    parser.add_argument('--ground-truth-answer', type=str, 
                       help='Ground truth answer for relevance scoring', default=None)
    args = parser.parse_args()
    print("Arguments parsed")
    # Load configuration
    config_manager = ConfigurationManager(args.config)
    print("Configuration loaded")
    
    # # Setup logging
    # setup_logging(config_manager.config)
    
    # Get implementation configuration
    impl_config = config_manager.get_implementation_config()
    print("Implementation configuration loaded")
    try:
        # Create implementation instance
        implementation = ImplementationFactory.create(
            impl_config['name'],
            impl_config['config']
        )
        print("Implementation created")
        
        # Process query
        result = implementation.process_query(args.query, args.ground_truth_answer)
        
        # Print answer
        print("\nAnswer:")
        print(result.get("answer", "No answer found"))
        
        # Print document sources if available
        if "document_sources" in result and result["document_sources"]:
            print("\nDocument Sources:")
            for source in result["document_sources"]:
                print(f" - {source}")
        
        # Print source relevance score if available
        if "source_relevance_score" in result:
            print("\nSource Relevance Score:")
            relevance = result["source_relevance_score"]
            print(f" - Average: {relevance['average']:.4f}")
            print(f" - Maximum: {relevance['maximum']:.4f}")
        
    finally:
        # Cleanup
        if 'implementation' in locals():
            implementation.cleanup()

if __name__ == '__main__':
    main() 