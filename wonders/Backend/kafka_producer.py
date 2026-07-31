"""
KafkaIngestionProducer — IngestionQueueProducer implementation. Publishes a
small JSON reference (job id + a path already on disk) to Kafka; it does NOT
ship the file's bytes through the queue. Backend/ingestion_worker.py's
consumer reads the actual file from `stored_path` and does the real
parsing/embedding work, in a separate process from the FastAPI app that
handled the upload — this is what lets many users upload concurrently
without each request blocking on (or competing for CPU during) embedding.

The kafka-python client connects lazily, on first publish() call, not at
construction time — Backend/deps.py builds this once per process via
lru_cache, and the app must be able to start even if Kafka isn't reachable
yet (mirrors the defensive pattern already used for the optional VLM/Ollama
tool — see Backend/conversation_service.py).
"""

from __future__ import annotations

import json


class KafkaIngestionProducer:
    def __init__(self, bootstrap_servers: str, topic: str):
        self._bootstrap_servers = bootstrap_servers
        self._topic = topic
        self._producer = None

    def _get_producer(self):
        if self._producer is None:
            from kafka import KafkaProducer

            self._producer = KafkaProducer(
                bootstrap_servers=self._bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                # kafka-python's defaults take ~30s to give up on an
                # unreachable broker, which would make every upload hang
                # that long before ingest_upload()'s try/except marks the
                # job failed. 5s/8s is still generous for a real local/LAN
                # broker (which typically responds in well under 100ms).
                request_timeout_ms=5000,
                max_block_ms=8000,
            )
        return self._producer

    def publish(
        self,
        job_id: int,
        conversation_id: int,
        user_id: int,
        file_name: str,
        stored_path: str,
    ) -> None:
        producer = self._get_producer()
        future = producer.send(
            self._topic,
            {
                "job_id": job_id,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "file_name": file_name,
                "stored_path": stored_path,
            },
        )
        # Blocks until the broker acks (or raises) so a queue-unreachable
        # error surfaces to the caller (ConversationService.ingest_upload)
        # instead of failing silently in a background I/O thread.
        future.get(timeout=8)
