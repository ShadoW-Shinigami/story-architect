"""
SQLite database configuration with async SQLAlchemy
"""

from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


# Database path
DATABASE_PATH = Path("data/story_architect.db")
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

# SQLite connection URL (async)
DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_PATH}"

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL debugging
    future=True,
)

# Async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


async def create_db_and_tables():
    """Create all database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database sessions.

    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Context manager for direct usage
class get_db_context:
    """
    Context manager for database sessions.

    Usage:
        async with get_db_context() as db:
            result = await db.execute(...)
    """
    def __init__(self):
        self.session = None

    async def __aenter__(self) -> AsyncSession:
        self.session = async_session_maker()
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.session.rollback()
        else:
            await self.session.commit()
        await self.session.close()
