"""
BROTEUS Visual Memory
======================

Few-shot learning via CLIP embedding storage.

When the user corrects a misclassification, we store the CLIP embedding
of that crop as a reference. On future detections, we compare against
stored references FIRST (cosine similarity), then fall back to text labels.

This implements error minimization: each correction makes BROTEUS smarter
without any retraining. Just embedding math.

Memory is persistent — saved to disk as a JSON + numpy file.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np

logger = logging.getLogger("broteus.memory")

MEMORY_DIR = Path("broteus_memory")
EMBEDDINGS_FILE = MEMORY_DIR / "embeddings.npy"
LABELS_FILE = MEMORY_DIR / "labels.json"


class VisualMemory:
    """Few-shot visual memory using CLIP embeddings.

    Stores (label, embedding) pairs. When a new detection comes in,
    compare its embedding against all stored references. If the best
    match exceeds a confidence threshold, use the stored label instead
    of CLIP's text-based classification.

    This means: the more you correct, the more accurate BROTEUS becomes.
    """

    def __init__(self, similarity_threshold: float = 0.80):
        """
        Args:
            similarity_threshold: Minimum cosine similarity to match
                a stored reference. Higher = stricter matching.
                0.80 is a good default for CLIP ViT-B-32.
        """
        self.similarity_threshold = similarity_threshold
        self.labels: List[str] = []           # Human-assigned labels
        self.embeddings: List[np.ndarray] = []  # Corresponding CLIP embeddings
        self._load()

    def store(self, label: str, embedding: np.ndarray):
        """Store a reference embedding for a label.

        Called when user corrects a classification.
        Multiple embeddings per label are allowed (different angles/lighting).
        """
        # Normalize embedding
        emb = embedding.flatten().astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        self.labels.append(label)
        self.embeddings.append(emb)
        self._save()
        logger.info(f"Visual memory: stored '{label}' ({len(self.embeddings)} total references)")

    def query(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Query the memory with a detection's CLIP embedding.

        Returns:
            (label, similarity) if a match is found above threshold
            (None, 0.0) if no match
        """
        if not self.embeddings:
            return None, 0.0

        # Normalize query
        query = embedding.flatten().astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-8)

        # Compute cosine similarity against all stored embeddings
        stored = np.stack(self.embeddings)  # (N, dim)
        similarities = stored @ query       # (N,)

        # Check ignores first with a LOWER threshold (easier to match)
        ignore_threshold = self.similarity_threshold - 0.15
        for i, (label, sim) in enumerate(zip(self.labels, similarities)):
            if label == '__IGNORE__' and sim >= ignore_threshold:
                return '__IGNORE__', float(sim)

        # Then check regular labels with normal threshold
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= self.similarity_threshold and self.labels[best_idx] != '__IGNORE__':
            return self.labels[best_idx], best_sim
        return None, best_sim

    def query_top_k(self, embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """Get top-k matches from memory.

        Useful for showing alternatives to the user.
        """
        if not self.embeddings:
            return []

        query = embedding.flatten().astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-8)

        stored = np.stack(self.embeddings)
        similarities = stored @ query

        top_indices = np.argsort(similarities)[::-1][:k]
        return [(self.labels[i], float(similarities[i])) for i in top_indices]

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        unique_labels = set(self.labels)
        label_counts = {}
        for l in self.labels:
            label_counts[l] = label_counts.get(l, 0) + 1
        return {
            "total_references": len(self.embeddings),
            "unique_labels": len(unique_labels),
            "labels": label_counts,
        }

    def remove_label(self, label: str) -> int:
        """Remove all references for a label. Returns count removed."""
        new_labels = []
        new_embeddings = []
        removed = 0
        for l, e in zip(self.labels, self.embeddings):
            if l.lower() == label.lower():
                removed += 1
            else:
                new_labels.append(l)
                new_embeddings.append(e)
        self.labels = new_labels
        self.embeddings = new_embeddings
        if removed > 0:
            self._save()
        return removed

    def clear(self):
        """Clear all memory."""
        self.labels = []
        self.embeddings = []
        self._save()
        logger.info("Visual memory cleared")

    def _save(self):
        """Persist memory to disk."""
        MEMORY_DIR.mkdir(exist_ok=True)
        with open(LABELS_FILE, 'w') as f:
            json.dump(self.labels, f)
        if self.embeddings:
            np.save(EMBEDDINGS_FILE, np.stack(self.embeddings))
        elif EMBEDDINGS_FILE.exists():
            EMBEDDINGS_FILE.unlink()
        logger.debug(f"Memory saved: {len(self.labels)} references")

    def _load(self):
        """Load memory from disk."""
        if LABELS_FILE.exists() and EMBEDDINGS_FILE.exists():
            try:
                with open(LABELS_FILE, 'r') as f:
                    self.labels = json.load(f)
                embs = np.load(EMBEDDINGS_FILE)
                self.embeddings = [embs[i] for i in range(len(embs))]
                logger.info(f"Visual memory loaded: {len(self.labels)} references")
            except Exception as e:
                logger.warning(f"Failed to load memory: {e}")
                self.labels = []
                self.embeddings = []
        else:
            logger.info("Visual memory: starting fresh (no saved data)")

    @property
    def count(self) -> int:
        return len(self.embeddings)
