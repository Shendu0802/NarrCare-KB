"""Unified error handling for NarrCare-KB."""


class KBErrorCode:
    KB_SERVICE_UNAVAILABLE = "KB_SERVICE_UNAVAILABLE"
    KB_INDEX_NOT_READY = "KB_INDEX_NOT_READY"
    KB_SCHEMA_INVALID = "KB_SCHEMA_INVALID"
    KB_RETRIEVAL_TIMEOUT = "KB_RETRIEVAL_TIMEOUT"
    KB_LLM_ERROR = "KB_LLM_ERROR"
    KB_EMBEDDING_ERROR = "KB_EMBEDDING_ERROR"
    KB_INGESTION_FAILED = "KB_INGESTION_FAILED"


class KBException(Exception):
    def __init__(self, error_code: str, detail: str = "", http_status: int = 500):
        self.error_code = error_code
        self.detail = detail
        self.http_status = http_status
        super().__init__(f"[{error_code}] {detail}")
