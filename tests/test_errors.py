from src.errors import KBException, KBErrorCode


class TestKBException:
    def test_error_codes_exist(self):
        assert KBErrorCode.KB_SERVICE_UNAVAILABLE == "KB_SERVICE_UNAVAILABLE"
        assert KBErrorCode.KB_INDEX_NOT_READY == "KB_INDEX_NOT_READY"
        assert KBErrorCode.KB_SCHEMA_INVALID == "KB_SCHEMA_INVALID"
        assert KBErrorCode.KB_RETRIEVAL_TIMEOUT == "KB_RETRIEVAL_TIMEOUT"
        assert KBErrorCode.KB_LLM_ERROR == "KB_LLM_ERROR"

    def test_exception_creation(self):
        exc = KBException(
            error_code=KBErrorCode.KB_INDEX_NOT_READY,
            detail="FAISS index file not found",
            http_status=503,
        )
        assert exc.error_code == "KB_INDEX_NOT_READY"
        assert exc.detail == "FAISS index file not found"
        assert exc.http_status == 503
        assert "KB_INDEX_NOT_READY" in str(exc)

    def test_exception_default_status(self):
        exc = KBException(error_code="KB_UNKNOWN")
        assert exc.http_status == 500
