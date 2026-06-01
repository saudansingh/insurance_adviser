import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from gemini_live import GeminiLive
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

# 🛠️ DB IMPORTS: Synchronized with Agent 1's memory tracking models
from database import async_session, get_or_create_user, load_memory, SessionSummary

# Load environment variables
load_dotenv()

# Configure logging: Keep at WARNING to avoid raw data packet stream spam
logging.basicConfig(level=logging.WARNING)
logging.getLogger("gemini_live").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL", "gemini-3.1-flash-live-preview")
ENVIRONMENT = os.getenv("ENVIRONMENT", "cloud")  # "local" or "cloud"

AGENT_ID = "insurance-adviser-agent"

# Audio Format Configurations
CHANNELS = 1
INPUT_RATE = 16000   # Mic input rate expected by Gemini Live
OUTPUT_RATE = 24000  # Speaker output rate sent back by Gemini Live
CHUNK_SIZE = 1024


async def save_session_summary(summary_id: int | None, user_id: int, conversation_text: str) -> int:
    """Create or update session summary row by ID. Returns row ID."""
    try:
        async with async_session() as session:
            if summary_id:
                result = await session.execute(
                    select(SessionSummary).where(SessionSummary.id == summary_id, SessionSummary.agent_id == AGENT_ID)
                )
                existing = result.scalar_one_or_none()
                if existing:
                    existing.summary = conversation_text
                    await session.commit()
                    logger.info(f"Updated summary row {summary_id} for user {user_id}")
                    return summary_id

            new_summary = SessionSummary(user_id=user_id, summary=conversation_text, agent_id=AGENT_ID)
            session.add(new_summary)
            await session.commit()
            await session.refresh(new_summary)
            logger.info(f"Created summary row {new_summary.id} for user {user_id}")
            return new_summary.id
    except Exception as e:
        logger.error(f"Failed to save session summary: {e}")
        return summary_id or 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate configuration on startup."""
    if not GEMINI_API_KEY:
        logger.warning("WARNING: GEMINI_API_KEY is not set. WebSocket endpoints will not work.")
    logger.info("Insurance Adviser API started successfully")
    yield

app = FastAPI(title="Insurance Adviser API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoints
@app.get("/")
async def root():
    return {"message": "Insurance Adviser API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Insurance Adviser API", "version": "1.0"}


# WebSocket endpoint for live audio chat
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio chat with Gemini Live API."""
    await websocket.accept()
    
    if not GEMINI_API_KEY:
        await websocket.send_json({"error": "GEMINI_API_KEY not configured"})
        await websocket.close()
        return

    user_email = websocket.query_params.get("email")
    if not user_email:
        await websocket.send_json({"error": "Missing 'email' connection parameter."})
        await websocket.close()
        return
    
    user_id = None
    session_summary_id = None         
    current_call_text_segments = []  

    async with async_session() as db_session:
        # 1. Initialize user and memory settings
        user_id = await get_or_create_user(user_email, db_session)
        past_chat_summary = await load_memory(user_id, AGENT_ID, db_session)

        system_instruction = "You are Ankur, a helpful and professional voice insurance adviser."
        if past_chat_summary and 'ChatContext object at' not in past_chat_summary:
            system_instruction += f"\n\nPREVIOUS CONVERSATION SUMMARY:\n{past_chat_summary}\n\nRemember to acknowledge this context naturally."

        audio_input_queue = asyncio.Queue()
        video_input_queue = asyncio.Queue()
        text_input_queue = asyncio.Queue()
        audio_playback_queue = asyncio.Queue()

        async def audio_output_callback(data):
            await audio_playback_queue.put(data)

        async def audio_interrupt_callback():
            while not audio_playback_queue.empty():
                try:
                    audio_playback_queue.get_nowait()
                    audio_playback_queue.task_done()
                except asyncio.QueueEmpty:
                    break

        gemini_client = GeminiLive(
            api_key=GEMINI_API_KEY, 
            model=MODEL, 
            input_sample_rate=INPUT_RATE,
            system_instruction=system_instruction
        )

        # Task 1: Inbound WebSockets reader
        async def handle_websocket_messages():
            try:
                while True:
                    data = await websocket.receive_bytes()
                    await audio_input_queue.put(data)
            except WebSocketDisconnect:
                logger.info("User disconnected websocket stream.")
            except Exception as e:
                logger.error(f"Inbound WS error: {e}")

        # Task 2: Outbound Audio writer
        async def send_audio_response():
            try:
                while True:
                    audio_data = await audio_playback_queue.get()
                    await websocket.send_bytes(audio_data)
                    audio_playback_queue.task_done()
            except Exception as e:
                logger.debug(f"Outbound audio stop: {e}")

        # Task 3: Background Gemini Core loop
        async def handle_gemini_session():
            nonlocal session_summary_id
            try:
                async for event in gemini_client.start_session(
                    audio_input_queue=audio_input_queue,
                    video_input_queue=video_input_queue,
                    text_input_queue=text_input_queue,
                    audio_output_callback=audio_output_callback,
                    audio_interrupt_callback=audio_interrupt_callback,
                ):
                    if event and isinstance(event, dict):
                        text_content = event.get("text") or event.get("content")
                        if text_content:
                            current_call_text_segments.append(text_content)
                            
                            # Real-time incremental save when sentences conclude
                            if text_content.endswith(('.', '?', '!')) and user_id:
                                conversation_text = "".join(current_call_text_segments).strip()
                                session_summary_id = await save_session_summary(
                                    session_summary_id, user_id, conversation_text
                                )
                            # ❌ NO WEBSOCKET TEXT DISPATCH EXISTS HERE anymore!
            except Exception as e:
                logger.error(f"Gemini session runtime error: {e}")

        # Execute concurrent worker tasks
        ws_task = asyncio.create_task(handle_websocket_messages())
        audio_task = asyncio.create_task(send_audio_response())
        gemini_task = asyncio.create_task(handle_gemini_session())

        try:
            # Keep execution bound until either the frontend drops or Gemini finishes
            await asyncio.wait([ws_task, gemini_task], return_when=asyncio.FIRST_COMPLETED)
        finally:
            # 🚀 CRITICAL FIX: Shield the database commit operation from disconnect cancellation drops
            if user_id and current_call_text_segments:
                conversation_text = "".join(current_call_text_segments).strip()
                if conversation_text:
                    print(f"DEBUG TRANSCRIPT TO SAVE:\n{conversation_text}") # Verify compilation in console logs
                    try:
                        logger.info(f"Connection ending. Executing SHIELDED database commit for user: {user_id}")
                        await asyncio.shield(save_session_summary(session_summary_id, user_id, conversation_text))
                    except Exception as save_err:
                        logger.error(f"Failed to save context block: {save_err}")

            # Clean up loops safely
            for task in [ws_task, audio_task, gemini_task]:
                if not task.done():
                    task.cancel()
            try:
                await websocket.close()
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
