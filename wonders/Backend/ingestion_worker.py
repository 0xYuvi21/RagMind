"""
Standalone Kafka consumer process for document ingestion — run separately
from the FastAPI app:

    uv run python -m Backend.ingestion_worker

Consumes {job_id, conversation_id, user_id, file_name, stored_path} messages
published by Backend/kafka_producer.py's KafkaIngestionProducer (see
Backend/conversation_service.py::ConversationService.ingest_upload()), reads
the already-saved file bytes from `stored_path`, and runs the exact same
VectorStoreFactory pipeline that endpoint used to run inline — reusing
Backend/deps.py's composition root directly (its functions are plain
callables, not FastAPI-specific; only their `Depends(...)` wrapping is)
so this worker and the web process are wired identically (same DB, same
Chroma persist dir, same embedding/OCR config).

Run one or more of these alongside the API; they share one consumer group
(settings.kafka_consumer_group) so Kafka spreads partitions across however
many worker processes are running, and none of them double-process the same
job.

No retry/backoff/dead-letter handling here — a failed job is marked FAILED
with its error message and left there; this is a known, documented
simplification (see CLAUDE.md), not a production-grade delivery guarantee.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from Backend import deps
from Backend.repositories import SqlAlchemyIngestionJobRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingestion_worker")


def _process_message(payload: dict, session_factory, vector_store_factory, audit_logger) -> None:
    session = session_factory()
    try:
        jobs = SqlAlchemyIngestionJobRepository(session)
        job = jobs.get(payload["job_id"])
        if job is None:
            logger.warning("Job %s no longer exists, skipping", payload["job_id"])
            return

        jobs.mark_processing(job.id)
        try:
            file_bytes = Path(payload["stored_path"]).read_bytes()
            store = vector_store_factory.get_or_create(str(payload["conversation_id"]))
            chunks_added = store.add_file_stream(
                file_content=file_bytes,
                file_name=payload["file_name"],
                source_tag="user_upload",
                image_output_dir=str(Path(payload["stored_path"]).parent),
            )
            jobs.mark_done(job.id, chunks_added)
            audit_logger.log_ingestion_event(
                payload["conversation_id"],
                payload["user_id"],
                job.id,
                payload["file_name"],
                status="done",
                chunks_added=chunks_added,
            )
            logger.info("Job %s done (%s chunks) — %s", job.id, chunks_added, payload["file_name"])
        except Exception as exc:
            jobs.mark_failed(job.id, str(exc))
            audit_logger.log_ingestion_event(
                payload["conversation_id"],
                payload["user_id"],
                job.id,
                payload["file_name"],
                status="failed",
                error_message=str(exc),
            )
            logger.exception("Job %s failed — %s", job.id, payload["file_name"])
    finally:
        session.close()


def run() -> None:
    from kafka import KafkaConsumer

    settings = deps.get_settings_cached()
    session_factory = deps.get_session_factory()
    vector_store_factory = deps.get_vector_store_factory()
    audit_logger = deps.get_audit_logger()

    consumer = KafkaConsumer(
        settings.kafka_ingestion_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )

    logger.info(
        "Ingestion worker listening — topic=%s group=%s bootstrap=%s",
        settings.kafka_ingestion_topic,
        settings.kafka_consumer_group,
        settings.kafka_bootstrap_servers,
    )
    for message in consumer:
        _process_message(message.value, session_factory, vector_store_factory, audit_logger)


if __name__ == "__main__":
    run()
