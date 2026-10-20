"""Web 层错误类型：路由抛 AppError，统一转成 {ok:false,error:{...}}。"""
from __future__ import annotations


class AppError(Exception):
    def __init__(self, message: str, status_code: int = 400, code: str = "bad_request"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


def bad(message: str, code: str = "bad_request") -> AppError:
    return AppError(message, 400, code)


def not_found(message: str) -> AppError:
    return AppError(message, 404, "not_found")


def conflict(message: str) -> AppError:
    return AppError(message, 409, "conflict")
