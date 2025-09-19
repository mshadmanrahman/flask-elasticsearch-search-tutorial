import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from elasticsearch import Elasticsearch


load_dotenv()


EMBEDDING_DIMS = 384
MIN_SCORE = 0.1
ELSER_MODEL_ID = os.environ.get("ELSER_MODEL_ID", "elser_model_2")


class Search:
    def __init__(self):
        elastic_api_key = os.environ.get("ELASTIC_API_KEY")
        if not elastic_api_key:
            raise RuntimeError(
                "ELASTIC_API_KEY is not set. Create a .env with ELASTIC_API_KEY=<your key>."
            )

        self.es = Elasticsearch(
            "https://3e5cd53f1eec4916ad122c933fa350c2.us-central1.gcp.cloud.es.io:443",
            api_key=elastic_api_key,
        )

        self._st_model = None  # Lazy-loaded SentenceTransformer

    # ---------------------- Index management ----------------------
    def create_index_dense(self) -> None:
        self.es.indices.delete(index="my_documents", ignore_unavailable=True)
        mapping: Dict[str, Any] = {
            "mappings": {
                "properties": {
                    "name": {"type": "text"},
                    "summary": {"type": "text"},
                    "content": {"type": "text"},
                    "category": {"type": "keyword"},
                    "updated_at": {"type": "date"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIMS,
                        "index": True,
                        "similarity": "cosine",
                    },
                    # Optional: ELSER sparse vector tokens
                    "elser_vector": {"type": "sparse_vector"},
                }
            }
        }
        self.es.indices.create(index="my_documents", mappings=mapping["mappings"]) 

    # ---------------------- Embeddings ----------------------
    def _ensure_st_model(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            # Force CPU to avoid meta-tensor device issues on some macOS/PyTorch builds
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def embed_text(self, text: str) -> List[float]:
        self._ensure_st_model()
        vector = self._st_model.encode(text or "")
        return vector.tolist()

    # ---------------------- ELSER tokens ----------------------
    def _infer_elser_tokens(self, text: str) -> Dict[str, float]:
        try:
            resp = self.es.ml.infer_trained_model(
                model_id=ELSER_MODEL_ID,
                docs=[{"text_field": text or ""}],
            )
        except Exception as e:
            # If the model isn't deployed, surface a helpful error
            raise RuntimeError(
                f"ELSER inference failed. Ensure model {ELSER_MODEL_ID} is deployed in your cluster."
            ) from e

        res = resp.get("inference_results", [{}])[0]
        # ELSER v2 returns tokens in predicted_value field
        tokens = res.get("predicted_value", {})
        if tokens:
            return tokens
        # Fallback to old format
        tokens = res.get("predicted_tokens") or res.get("entities") or []
        # tokens may be list of {token, weight}
        token_map: Dict[str, float] = {}
        for t in tokens:
            token = t.get("token") or t.get("ngram")
            weight = float(t.get("weight", 0.0))
            if token:
                token_map[token] = weight
        return token_map

    # ---------------------- Ingestion ----------------------
    def insert_document(self, document: Dict[str, Any]):
        return self.es.index(index="my_documents", document=document)

    def insert_documents(self, documents: List[Dict[str, Any]]):
        operations: List[Dict[str, Any]] = []
        for document in documents:
            operations.append({"index": {"_index": "my_documents"}})
            operations.append(document)
        return self.es.bulk(operations=operations)

    def reindex_dense_with_embeddings(self) -> Dict[str, Any]:
        self.create_index_dense()
        with open("data.json", "rt") as f:
            documents: List[Dict[str, Any]] = json.loads(f.read())

        for d in documents:
            # Use summary primarily; fallback to content/name
            basis = d.get("summary") or d.get("content") or d.get("name", "")
            d["embedding"] = self.embed_text(basis)

        return self.insert_documents(documents)

    def reindex_dense_and_elser(self) -> Dict[str, Any]:
        self.create_index_dense()
        with open("data.json", "rt") as f:
            documents: List[Dict[str, Any]] = json.loads(f.read())
        for d in documents:
            basis = d.get("summary") or d.get("content") or d.get("name", "")
            d["embedding"] = self.embed_text(basis)
            d["elser_vector"] = self._infer_elser_tokens(basis)
        return self.insert_documents(documents)

    # ---------------------- Retrieval ----------------------
    def search_bm25(self, *, query: str, filters: Dict[str, Any], size: int, from_: int) -> Dict[str, Any]:
        if query:
            must = {"multi_match": {"query": query, "fields": ["name", "summary", "content"]}}
        else:
            must = {"match_all": {}}
        return self.es.search(
            index="my_documents",
            query={"bool": {"must": must, **filters}},
            aggs={
                "category-agg": {"terms": {"field": "category.keyword"}},
                "year-agg": {
                    "date_histogram": {"field": "updated_at", "calendar_interval": "year", "format": "yyyy"}
                },
            },
            size=size,
            from_=from_,
            min_score=MIN_SCORE,
        )

    def search_knn(self, *, query: str, size: int, from_: int, k: int = 10, num_candidates: int = 50,
                    filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        filters = filters or {}
        qv = self.embed_text(query)
        # Combine vector search with optional filters by wrapping knn + filter query
        return self.es.search(
            index="my_documents",
            knn={
                "field": "embedding",
                "query_vector": qv,
                "k": k,
                "num_candidates": num_candidates,
                # filter can be supplied within knn in recent versions; also keep top-level query for safety
                "filter": filters.get("filter") if filters else None,
            },
            query={"bool": {"filter": filters.get("filter", [])}},
            size=size,
            from_=from_,
            min_score=MIN_SCORE,
        )

    def search_elser(self, *, query: str, size: int, from_: int, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        filters = filters or {}
        # text_expansion uses model_id server-side and matches against pre-indexed tokens
        q = {
            "bool": {
                "must": {
                    "text_expansion": {
                        "elser_vector": {"model_id": ELSER_MODEL_ID, "model_text": query}
                    }
                },
                **filters,
            }
        }
        try:
            return self.es.search(index="my_documents", query=q, size=size, from_=from_, min_score=MIN_SCORE)
        except Exception:
            # Gracefully degrade if ELSER model is not deployed
            return {"hits": {"total": {"value": 0}, "hits": []}}

    def _rrf_fuse(self, hits_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        # Reciprocal Rank Fusion in Python
        doc_to_score: Dict[str, float] = {}
        doc_to_hit: Dict[str, Dict[str, Any]] = {}
        for hits in hits_lists:
            for rank, h in enumerate(hits, start=1):
                doc_id = h.get("_id")
                if not doc_id:
                    continue
                doc_to_hit.setdefault(doc_id, h)
                doc_to_score[doc_id] = doc_to_score.get(doc_id, 0.0) + 1.0 / (k + rank)
        # sort by fused score desc
        fused_ids = sorted(doc_to_score.keys(), key=lambda d: doc_to_score[d], reverse=True)
        fused_hits = [doc_to_hit[i] for i in fused_ids]
        # attach fused score for display if needed
        for h in fused_hits:
            h.setdefault("_score", 0.0)
            h["_score"] = float(h["_score"])  # ensure numeric
        return fused_hits

    def search_hybrid_rrf_lexical_elser(self, *, query: str, size: int, filters: Dict[str, Any]) -> Dict[str, Any]:
        bm25 = self.search_bm25(query=query, filters=filters, size=size, from_=0)
        elser = self.search_elser(query=query, filters=filters, size=size, from_=0)
        fused_hits = self._rrf_fuse([bm25["hits"]["hits"], elser["hits"]["hits"]])
        # Build a response-like dict
        body: Dict[str, Any] = {
            "hits": {"total": {"value": len(fused_hits)}, "hits": fused_hits[:size]},
        }
        # carry over aggs from bm25 if present
        if "aggregations" in bm25:
            body["aggregations"] = bm25["aggregations"]
        return body

    def search_hybrid_rrf(self, *, query: str, size: int, from_: int, k: int = 10, num_candidates: int = 50,
                           filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        filters = filters or {}
        qv = self.embed_text(query)
        rank_window_size = max(size, 50)
        retriever = {
            "rrf": {
                "retrievers": [
                    {"standard": {"query": {"bool": {"must": {"multi_match": {"query": query, "fields": ["name", "summary", "content"]}}, **filters}}}},
                    {"knn": {"field": "embedding", "query_vector": qv, "k": k, "num_candidates": num_candidates}},
                ],
                "rank_window_size": rank_window_size,
                "rank_constant": 20,
            }
        }
        return self.es.search(index="my_documents", retriever=retriever, size=size, from_=from_)

    # Backwards compatibility
    def search(self, **query_args):
        return self.es.search(index="my_documents", **query_args)

    def retrieve_document(self, id):
        return self.es.get(index="my_documents", id=id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Utilities for my_documents index")
    parser.add_argument("action", choices=["reindex-dense", "reindex-dense-elser"], help="Action to perform")
    args = parser.parse_args()

    s = Search()
    if args.action == "reindex-dense":
        res = s.reindex_dense_with_embeddings()
        print("Reindexed with embeddings. items:", res.get("items") and len(res["items"]))
    elif args.action == "reindex-dense-elser":
        res = s.reindex_dense_and_elser()
        print("Reindexed with embeddings + ELSER. items:", res.get("items") and len(res["items"]))
