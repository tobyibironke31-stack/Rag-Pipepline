from llama_index.core.retrievers import BaseRetriever

class HybridRetriever(BaseRetriever):
    """Combines dense semantic search with BM25 sparse keyword retrieval."""

    def __init__(self, vector_retriever, keyword_retriever, top_k=2):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.top_k = top_k
        super().__init__()

    def _retrieve(self, query_bundle, **kwargs):
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        keyword_nodes = self.keyword_retriever.retrieve(query_bundle)

        # Merge nodes and deduplicate by node_id
        unique_nodes = {}
        for node in list(vector_nodes) + list(keyword_nodes):
            if node.node_id not in unique_nodes:
                unique_nodes[node.node_id] = node

        # Sort combined results descending by score
        sorted_nodes = sorted(
            unique_nodes.values(),
            key=lambda x: x.score if hasattr(x, "score") and x.score is not None else 0.0,
            reverse=True,
        )
        return sorted_nodes[: self.top_k]
