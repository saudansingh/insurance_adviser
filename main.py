import asyncio
import logging
import os
from dotenv import load_dotenv
from gemini_live import GeminiLive
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

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

# Audio Format Configurations (for reference)
CHANNELS = 1
INPUT_RATE = 16000   # Mic input rate expected by Gemini Live
OUTPUT_RATE = 24000  # Speaker output rate sent back by Gemini Live
CHUNK_SIZE = 1024

# Initialize FastAPI app
app = FastAPI(title="Insurance Adviser API")

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
            # Read raw PCM data from the microphone without blocking the main loop
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
            # Write raw PCM data out to hardware speakers
            await asyncio.to_thread(output_stream.write, data)
            audio_playback_queue.task_done()
        except Exception as e:
            logger.error(f"Error playing audio: {e}")


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Insurance Adviser API"}

# WebSocket endpoint for live audio chat
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio chat with Gemini Live API."""
    await websocket.accept()
    
    if not GEMINI_API_KEY:
        await websocket.send_json({"error": "GEMINI_API_KEY not configured"})
        await websocket.close()
        return
    
    try:
        # Live API Stream Communication Queues
        audio_input_queue = asyncio.Queue()
        video_input_queue = asyncio.Queue()
        text_input_queue = asyncio.Queue()
        audio_playback_queue = asyncio.Queue()

        # Audio callbacks
        async def audio_output_callback(data):
            """Send audio response back to client."""
            await audio_playback_queue.put(data)

        async def audio_interrupt_callback():
            """Handle interruptions."""
            while not audio_playback_queue.empty():
                try:
                    audio_playback_queue.get_nowait()
                    audio_playback_queue.task_done()
                except asyncio.QueueEmpty:
                    break

        # Initialize Gemini Live Engine
        gemini_client = GeminiLive(
            api_key=GEMINI_API_KEY, 
            model=MODEL, 
            input_sample_rate=INPUT_RATE
        )

        # Handle incoming WebSocket messages
        async def handle_websocket_messages():
            try:
                while True:
                    data = await websocket.receive_bytes()
                    await audio_input_queue.put(data)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

        # Handle outgoing audio
        async def send_audio_response():
            try:
                while True:
                    audio_data = await audio_playback_queue.get()
                    await websocket.send_bytes(audio_data)
                    audio_playback_queue.task_done()
            except Exception as e:
                logger.error(f"Error sending audio: {e}")

        # Start tasks
        ws_task = asyncio.create_task(handle_websocket_messages())
        audio_task = asyncio.create_task(send_audio_response())

        # Run Gemini session
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
                    await websocket.send_json({"text": text_content})

    except Exception as e:
        logger.error(f"WebSocket session error: {e}")
    finally:
        try:
            await websocket.close()
        except:
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

    # Initialize PyAudio Engine
    p = pyaudio.PyAudio()
    AUDIO_FORMAT = pyaudio.paInt16

    # Open Hardware Microphone Stream (16kHz)
    input_stream = p.open(
        format=AUDIO_FORMAT,
        channels=CHANNELS,
        rate=INPUT_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    # Open Hardware Speaker Stream (24kHz)
    output_stream = p.open(
        format=AUDIO_FORMAT,
        channels=CHANNELS,
        rate=OUTPUT_RATE,
        output=True
    )

    # Live API Stream Communication Queues
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
        # Run with microphone
        try:
            asyncio.run(main_local())
        except KeyboardInterrupt:
            print("\nShutdown...")
    else:
        # Run FastAPI server for Cloud Run
        import uvicorn
        port = int(os.getenv("PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port)
        print("\nSession stopped by user.")