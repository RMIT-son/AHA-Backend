import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.logging import setup_logger

logger = setup_logger()

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Unhandled exception: {e}", exc_info=True)
            raise
        process_time = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} | "
            f"Status: {response.status_code} | "
            f"Time: {process_time:.2f}s"
        )
        return response
