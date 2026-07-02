"""POST /index/rebuild endpoint."""
from fastapi import APIRouter
from src.config import settings
from src.errors import KBException, KBErrorCode

router = APIRouter(prefix="/index", tags=["indexing"])


@router.post("/rebuild")
async def rebuild_index(full: bool = True):
    try:
        from src.db.connection import Database
        from src.indexing.text_index import TextIndex

        db = Database(settings.db_path)
        db.initialize()

        text_index = TextIndex(db)
        text_index.rebuild()

        from src.indexing.embedder import Embedder
        from src.indexing.vector_index import VectorIndex
        from src.db.repository import KnowledgeUnitRepository

        repo = KnowledgeUnitRepository(db)
        active = repo.get_all_active()

        vec_index_size = 0
        if active:
            embedder = Embedder(
                model_path=f"{settings.models_dir}/{settings.embedding_model}",
                device=settings.embedding_device,
            )
            if not embedder.load():
                raise KBException(
                    error_code=KBErrorCode.KB_EMBEDDING_ERROR,
                    detail="Failed to load embedding model", http_status=500,
                )

            texts = [u["text"] for u in active]
            ids = [u["id"] for u in active]
            embeddings = embedder.encode(texts, batch_size=settings.embedding_batch_size)

            vec_index = VectorIndex(dim=embeddings.shape[1], index_path=settings.faiss_index_path)
            vec_index.build(ids, embeddings)
            vec_index.save()
            vec_index_size = vec_index.size

        db.close()
        return {"status": "done", "faiss_size": vec_index_size, "fts5_rebuilt": True, "chunk_count": len(active)}
    except KBException:
        raise
    except Exception as e:
        raise KBException(error_code=KBErrorCode.KB_EMBEDDING_ERROR, detail=str(e), http_status=500)
