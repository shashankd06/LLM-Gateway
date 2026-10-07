"""Expected gateway failures carry only safe, client-visible messages."""


class GatewayError(Exception):
    def __init__(
        self, status_code: int, message: str, code: str, retry_after: int | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code
        self.retry_after = retry_after
