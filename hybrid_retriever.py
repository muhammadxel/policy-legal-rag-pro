import re
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer for BM25 keyword indexing."""
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """Combines ChromaDB semantic vector search with BM25 lexical search via Reciprocal Rank Fusion (RRF)."""

    def __init__(self, collection):
        self.collection = collection
        self.bm25 = None
        self.corpus_ids: list[str] = []
        self.corpus_docs: list[str] = []
        self.corpus_metas: list[dict] = []
        self.refresh_bm25_index()

    def refresh_bm25_index(self):
        """Sync in-memory BM25 keyword index with all documents currently stored in ChromaDB."""
        data = self.collection.get(include=["documents", "metadatas"])
        self.corpus_ids = data.get("ids", [])
        self.corpus_docs = data.get("documents", [])
        self.corpus_metas = data.get("metadatas", [])

        if self.corpus_docs:
            tokenized_corpus = [_tokenize(doc) for doc in self.corpus_docs]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def add_records(self, records: list[dict]):
        """Upsert metadata-tagged chunks into ChromaDB and rebuild the BM25 index."""
        if not records:
            return

        ids = [r["id"] for r in records]
        documents = [r["text"] for r in records]
        metadatas = [r["metadata"] for r in records]

        # Batch upsert in chunks of 100 to avoid SQLite variable limits
        batch_size = 100
        for i in range(0, len(records), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        self.refresh_bm25_index()

    def delete_document(self, source_filename: str):
        """Remove all chunks belonging to a specific document filename."""
        self.collection.delete(where={"source": source_filename})
        self.refresh_bm25_index()

    def clear_all(self):
        """Wipe all indexed documents from the vector store."""
        existing = self.collection.get()["ids"]
        if existing:
            self.collection.delete(ids=existing)
        self.refresh_bm25_index()

    def get_indexed_sources(self) -> dict[str, int]:
        """Return a dictionary mapping each indexed filename to its chunk count."""
        counts: dict[str, int] = {}
        for meta in self.corpus_metas:
            src = meta.get("source", "Unknown")
            counts[src] = counts.get(src, 0) + 1
        return counts

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        Execute hybrid retrieval (Vector + BM25) and fuse rankings using RRF.
        Optionally filter results to a specific document filename.
        """
        if not self.corpus_ids:
            return []

        candidate_limit = min(len(self.corpus_ids), top_k * 3)
        where_clause = {"source": source_filter} if source_filter else None

        # 1. Dense Semantic Vector Search (ChromaDB)
        vector_results = self.collection.query(
            query_texts=[query],
            n_results=candidate_limit,
            where=where_clause,
            include=["documents", "metadatas"],
        )

        vec_ids = vector_results["ids"][0] if vector_results["ids"] else []
        vec_docs = vector_results["documents"][0] if vector_results["documents"] else []
        vec_metas = vector_results["metadatas"][0] if vector_results["metadatas"] else []

        doc_lookup = {}
        for doc_id, text, meta in zip(vec_ids, vec_docs, vec_metas):
            doc_lookup[doc_id] = {"id": doc_id, "text": text, "metadata": meta}

        # 2. Sparse Lexical Keyword Search (BM25)
        bm25_ranked_ids = []
        if self.bm25 is not None:
            tokenized_query = _tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)

            scored_indices = sorted(
                range(len(scores)), key=lambda idx: scores[idx], reverse=True
            )

            for idx in scored_indices:
                if scores[idx] <= 0:
                    continue
                meta = self.corpus_metas[idx]
                if source_filter and meta.get("source") != source_filter:
                    continue
                doc_id = self.corpus_ids[idx]
                bm25_ranked_ids.append(doc_id)
                if doc_id not in doc_lookup:
                    doc_lookup[doc_id] = {
                        "id": doc_id,
                        "text": self.corpus_docs[idx],
                        "metadata": meta,
                    }
                if len(bm25_ranked_ids) >= candidate_limit:
                    break

        # 3. Reciprocal Rank Fusion (RRF): score = sum(1 / (k + rank))
        rrf_scores: dict[str, float] = {}
        for rank, doc_id in enumerate(vec_ids, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

        for rank, doc_id in enumerate(bm25_ranked_ids, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

        # Sort fused candidates by combined RRF score descending
        fused_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[:top_k]

        final_results = []
        for doc_id in fused_ids:
            item = doc_lookup[doc_id]
            item["rrf_score"] = round(rrf_scores[doc_id], 4)
            final_results.append(item)

        return final_results