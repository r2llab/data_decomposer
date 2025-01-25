from symphony.embeddings import Embedder

# Initialize
embedder = Embedder()

# Embed text
text_embedding = embedder.embed_text("Your text here")
print(text_embedding)

# # Embed tabular data
# import pandas as pd
# df = pd.DataFrame(...)
# tabular_embedding = embedder.embed_tabular(df)

# # Compute similarities
# similarities = embedder.compute_similarity(embedding1, embedding2)