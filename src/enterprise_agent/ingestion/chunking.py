from hashlib import sha256

from enterprise_agent.models import ChunkSettings, Document, DocumentChunk


def chunk_document(document: Document, settings: ChunkSettings) -> list[DocumentChunk]:
    text = document.content
    if not text.strip():
        return []
    chunks = []
    for start in range(0, len(text), settings.size - settings.overlap):
        end = min(start + settings.size, len(text))
        content = text[start:end]
        if content.strip():
            digest = sha256(content.encode()).hexdigest()
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.document_id}:{start}:{digest}",
                    document_id=document.document_id,
                    position=len(chunks),
                    start_offset=start,
                    end_offset=end,
                    content=content,
                )
            )
        if end == len(text):
            break
    return chunks
