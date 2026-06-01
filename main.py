import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from gemini_live import GeminiLive
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# 🛠️ DB IMPORTS: Maps user context tracking engines seamlessly
from database import async_session, get_or_create_user, load_memory, save_summary

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

# =========================================================================
# 🎯 TARGET IDENTIFIER: Unique signature for the Insurance Agent
# =========================================================================
AGENT_ID = "insurance-adviser-agent"

# Audio Format Configurations (for reference)
CHANNELS = 1
INPUT_RATE = 16000   # Mic input rate expected by Gemini Live
OUTPUT_RATE = 24000  # Speaker output rate sent back by Gemini Live
CHUNK_SIZE = 1024

# Modern Lifespan utility replaces deprecated @app.on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate configuration on startup."""
    if not GEMINI_API_KEY:
        logger.warning("WARNING: GEMINI_API_KEY is not set. WebSocket endpoints will not work.")
    logger.info("Insurance Adviser API started successfully")
    yield

# Initialize FastAPI app with lifespan handler
app = FastAPI(title="Insurance Adviser API", lifespan=lifespan)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def mic_audio_recorder(audio_input_queue: asyncio.Queue, input_stream):
    """Continuously captures microphone input and feeds it into Gemini's input queue."""
    while True:
        try:
            data = await asyncio.to_thread(input_stream.read, CHUNK_SIZE, False)
            if data:
                await audio_input_queue.put(data)
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            await asyncio.sleep(0.01)


async def speaker_audio_player(audio_playback_queue: asyncio.Queue, output_stream):
    """Continuously processes Gemini's outbound audio queue and plays it through speakers."""
    while True:
        try:
            data = await audio_playback_queue.get()
            await asyncio.to_thread(output_stream.write, data)
            audio_playback_queue.task_done()
        except Exception as e:
            logger.error(f"Error playing audio: {e}")


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
    
    ws_task = None
    audio_task = None
    user_id = None
    current_call_text_segments = []  # Tracks live transcript tokens for summarizing

    # Wrap entire connection context in an active database connection
    async with async_session() as db_session:
        try:
            # 1. Look up user integer ID or generate a row if they are new
            user_id = await get_or_create_user(user_email, db_session)

            # 2. Extract context history exclusively belonging to THIS user and agent
            past_chat_summary = await load_memory(user_id, AGENT_ID, db_session)

            # 3. Construct dynamic system instructions containing isolation memory context
            system_instruction = "You are a helpful and professional voice insurance adviser."
            if past_chat_summary:
                system_instruction += f" Context from your past interactions with this user: {past_chat_summary}"

            # Live API Stream Communication Queues
            audio_input_queue = asyncio.Queue()
            video_input_queue = asyncio.Queue()
            text_input_queue = asyncio.Queue()
            audio_playback_queue = asyncio.Queue()

            # Audio callbacks
            async def audio_output_callback(data):
                await audio_playback_queue.put(data)

            async def audio_interrupt_callback():
                while not audio_playback_queue.empty():
                    try:
                        audio_playback_queue.get_nowait()
                        audio_playback_queue.task_done()
                    except asyncio.QueueEmpty:
                        break

            # Initialize Gemini Live Engine with the historical memory injected
            gemini_client = GeminiLive(
                api_key=GEMINI_API_KEY, 
                model=MODEL, 
                input_sample_rate=INPUT_RATE,
                system_instruction=system_instruction
            )

            # Handle incoming WebSocket messages
            async def handle_websocket_messages():
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        await audio_input_queue.put(data)
                except (WebSocketDisconnect, Exception):
                    pass

            # Handle outgoing audio
            async def send_audio_response():
                try:
                    while True:
                        audio_data = await audio_playback_queue.get()
                        await websocket.send_bytes(audio_data)
                        audio_playback_queue.task_done()
                except (WebSocketDisconnect, Exception):
                    pass

            # Start background tasks
            ws_task = asyncio.create_task(handle_websocket_messages())
            audio_task = asyncio.create_task(send_audio_response())

            # Run Gemini streaming session loop
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
                        # ✅ KEPT: Append response strings to build the background transcript block
                        current_call_text_segments.append(text_content)
                        
                        # 🛠️ REMOVED: await websocket.send_json({"text": text_content})
                        # Dropping this line ensures the frontend UI receives no text frames, keeping it clean!

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected naturally for user {user_email}")
        except Exception as e:
            logger.error(f"WebSocket session error: {e}")
        finally:
            # Summary persistence is executed inside the finally block.
            if user_id and current_call_text_segments:
                summary_text = "".join(current_call_text_segments).strip()
                if summary_text:
                    try:
                        logger.info(f"Connection ending. Saving Insurance record summary for User ID: {user_id}")
                        # 🛠️ CHANGED: Swapped summary_text[:400] to full summary_text to store the entire conversation block inside the DB
                        await save_summary(user_id, summary_text, AGENT_ID, db_session)
                    except Exception as save_err:
                        logger.error(f"Failed to auto-save summary context block: {save_err}")

            # Clean up pending loops safely
            if ws_task and not ws_task.done():
                ws_task.cancel()
            if audio_task and not audio_task.done():
                audio_task.cancel()
            try:
                await websocket.close()
            except Exception:
                pass


# Local testing with microphone (optional, requires pyaudio)
async def main_local():
    """Run locally with microphone audio input."""
    try:
        import pyaudio
    except ImportError:
        print("PyAudio not installed. Install with: pip install pyaudio")
        return
    
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY is missing from your .env file.")
        return

    p = pyaudio.PyAudio()
    AUDIO_FORMAT = pyaudio.paInt16

    input_stream = p.open(
        format=AUDIO_FORMAT,
        channels=CHANNELS,
        rate=INPUT_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    output_stream = p.open(
        format=AUDIO_FORMAT,
        channels=CHANNELS,
        rate=OUTPUT_RATE,
        output=True
    )

    audio_input_queue = asyncio.Queue()
    video_input_queue = asyncio.Queue()
    text_input_queue = asyncio.Queue()
    audio_playback_queue = asyncio.Queue()

    async def audio_output_callback(data):
        await audio_playback_queue.put(data)

    async def audio_interrupt_callback():
        print("\n[Interrupted] Clearing agent response queue...")
        while not audio_playback_queue.empty():
            try:
                audio_playback_queue.get_nowait()
                audio_playback_queue.task_done()
            except asyncio.QueueEmpty:
                break

    gemini_client = GeminiLive(
        api_key=GEMINI_API_KEY, model=MODEL, input_sample_rate=INPUT_RATE
    )

    record_task = asyncio.create_task(mic_audio_recorder(audio_input_queue, input_stream))
    play_task = asyncio.create_task(speaker_audio_player(audio_playback_queue, output_stream))

    print("🎙️  Voice Session Started! Start talking...")

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
                    print(text_content, end="", flush=True)
    except Exception as e:
        import traceback
        print(f"\nSession closed: {traceback.format_exc()}")
    finally:
        record_task.cancel()
        play_task.cancel()
        try:
            input_stream.stop_stream()
            input_stream.close()
            output_stream.stop_stream()
            output_stream.close()
            p.terminate()
        except Exception:
            pass
        print("\n🔒 Audio engine disconnected.")


if __name__ == "__main__":
    if ENVIRONMENT == "local":
        try:
            asyncio.run(main_local())
        except KeyboardInterrupt:
            print("\nShutdown...")
    else:
        # Run FastAPI server for Cloud Run
        import uvicorn
        port = int(os.getenv("PORT", 8080))
        uvicorn.run(app, host="0.0.0.0", port=port)
        print("\nSession stopped by user.")
