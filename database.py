import os
import logging
from sqlalchemy import String, Integer, Text, ForeignKey, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

logger = logging.getLogger(__name__)

# 1. Grab the Database connection URL from your GCP Environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

# Quick safeguard check for PostgreSQL async driver requirement
if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

if not DATABASE_URL:
    logger.error("CRITICAL: DATABASE_URL environment variable is missing!")

# 2. Setup the Async Database Engine and Session Factory
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,  # Automatically drops and recreates stale/dead connections
    pool_size=5,
    max_overflow=10
)

async_session = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# 3. Define the Database Base and Table Schemas
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)

class SessionSummary(Base):
    __tablename__ = "session_summaries"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    agent_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)


# 4. Core Async Database Helper Functions required by main.py

async def init_db():
    """Initializes and builds missing tables in the database target on startup."""
    if not DATABASE_URL:
        raise ValueError("Cannot initialize DB: DATABASE_URL is not set.")
    
    async with engine.begin() as conn:
        # Creates tables safely only if they do not already exist in your database
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schemas checked and initialized successfully.")


async def get_or_create_user(email: str, session: AsyncSession) -> int:
    """Finds a user by email, or provisions a new user row if not found. Returns user ID."""
    clean_email = email.strip().lower()
    try:
        # Check if user exists
        stmt = select(User).where(User.email == clean_email)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        
        if user:
            return user.id
            
        # Create user if they don't exist
        logger.info(f"Creating record for new user: {clean_email}")
        new_user = User(email=clean_email)
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)
        return new_user.id
    except Exception as e:
        await session.rollback()
        logger.error(f"Error in get_or_create_user for {clean_email}: {e}")
        raise e


async def load_memory(user_id: int, agent_id: str, session: AsyncSession) -> str | None:
    """Fetches the most recent conversation summary to give the agent context memory."""
    try:
        # Pull the latest summary row entry saved for this specific user and agent ID context
        stmt = (
            select(SessionSummary)
            .where(SessionSummary.user_id == user_id, SessionSummary.agent_id == agent_id)
            .order_by(SessionSummary.id.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        latest_record = result.scalar_one_or_none()
        
        if latest_record:
            logger.info(f"Loaded historic context memory block for user ID: {user_id}")
            return latest_record.summary
            
        logger.info(f"No existing historic session memory found for user ID: {user_id}")
        return None
    except Exception as e:
        logger.error(f"Failed to fetch conversation history context memory block: {e}")
        return None
