from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db.models import RequestLog


def estimate_cost(input_tokens: int, output_tokens: int, settings: Settings) -> Decimal:
    """Use Decimal to avoid accumulating binary floating-point errors in money totals."""
    return (
        input_tokens * settings.input_cost_per_1k + output_tokens * settings.output_cost_per_1k
    ) / Decimal(1000)


async def save_log(session: AsyncSession, log: RequestLog) -> None:
    session.add(log)
    await session.commit()
