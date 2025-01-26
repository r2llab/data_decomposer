# Sample data for testing the Symphony query engine

sample_data = [
    {
        "type": "document",
        "content": "The quarterly financial report shows a 15% increase in revenue. Growth was primarily driven by expansion in Asian markets.",
        "metadata": {"category": "finance", "date": "2024-01-15", "department": "finance"}
    },
    {
        "type": "person",
        "content": "Sarah Chen is the Head of AI Research, leading projects in natural language processing and computer vision.",
        "metadata": {"role": "head_of_research", "department": "ai", "location": "San Francisco"}
    },
    {
        "type": "product",
        "content": "The Symphony 2.0 platform features improved query understanding and real-time analytics capabilities.",
        "metadata": {"version": "2.0", "release_date": "2024-02-01", "category": "software"}
    },
    {
        "type": "document",
        "content": "Team meeting notes: Discussed upcoming product launch timeline. Marketing campaign to start in March. Budget approved.",
        "metadata": {"category": "meeting_notes", "date": "2024-02-10", "department": "marketing"}
    },
    {
        "type": "person",
        "content": "Alex Rodriguez is a Senior Software Engineer specializing in distributed systems and cloud architecture.",
        "metadata": {"role": "senior_engineer", "department": "engineering", "location": "New York"}
    },
    {
        "type": "support_ticket",
        "content": "User reported intermittent connection issues with the API endpoint. Investigation shows potential network latency problems.",
        "metadata": {"priority": "high", "status": "open", "category": "technical"}
    }
]

# Usage example:
if __name__ == "__main__":
    from symphony.core import Pipeline
    from pathlib import Path

    # Initialize pipeline
    pipeline = Pipeline()

    # Index the data
    texts = [item["content"] for item in sample_data]
    pipeline.index_data(sample_data, texts)

    # Test some queries
    test_queries = [
        "Who is the head of AI research?",
        # "What are the latest financial results?",
        # "Tell me about Symphony 2.0 features",
        # "Are there any high priority support issues?",
        # "What's happening with the marketing campaign?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = pipeline.run_query(query)
        print(f"Result: {result}")

    # Optionally save the index
    pipeline.discovery.save_index(Path("./sample_index"))