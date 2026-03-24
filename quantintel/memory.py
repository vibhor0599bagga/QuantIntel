# quantintel/memory.py
from rank_bm25 import BM25Okapi
from typing import List, Tuple
import re


class FinancialSituationMemory:
    """BM25-based memory. No API calls, works offline."""

    def __init__(self, name: str, config: dict = None):
        self.name          = name
        self.documents:    List[str] = []
        self.recommendations: List[str] = []
        self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _rebuild_index(self):
        if self.documents:
            self.bm25 = BM25Okapi([self._tokenize(d) for d in self.documents])
        else:
            self.bm25 = None

    def add_situations(self, situations_and_advice: List[Tuple[str, str]]):
        for situation, recommendation in situations_and_advice:
            self.documents.append(situation)
            self.recommendations.append(recommendation)
        self._rebuild_index()

    def get_memories(self, current_situation: str, n_matches: int = 1) -> List[dict]:
        if not self.documents or self.bm25 is None:
            return []
        scores     = self.bm25.get_scores(self._tokenize(current_situation))
        top_idx    = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_matches]
        max_score  = max(scores) if max(scores) > 0 else 1
        return [
            {
                "matched_situation": self.documents[i],
                "recommendation":    self.recommendations[i],
                "similarity_score":  scores[i] / max_score,
            }
            for i in top_idx
        ]

    def clear(self):
        self.documents = []
        self.recommendations = []
        self.bm25 = None
