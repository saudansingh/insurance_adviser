import asyncio
import logging
import os
from dotenv import load_dotenv
import pyaudio
from gemini_live import GeminiLive

# Load environment variables
load_dotenv()

# Configure logging: Keep at WARNING to avoid raw data packet stream spam
logging.basicConfig(level=logging.WARNING)
logging.getLogger("gemini_live").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL", "gemini-3.1-flash-live-preview")

# Audio Format Configurations
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000   # Mic input rate expected by Gemini Live
OUTPUT_RATE = 24000  # Speaker output rate sent back by Gemini Live
CHUNK_SIZE = 1024


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


async def main():
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY is missing from your .env file.")
        return

    # Initialize PyAudio Engine
    p = pyaudio.PyAudio()

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
    
    # Internal queue to sequentialize outbound speaker playback
    audio_playback_queue = asyncio.Queue()

    # Audio Callback definitions bound to the stream handler
    async def audio_output_callback(data):
        """Triggered when Gemini sends back raw speech packets."""
        await audio_playback_queue.put(data)

    async def audio_interrupt_callback():
        """Triggered automatically if you speak over Gemini while it is talking."""
        print("\n[Interrupted] Clearing agent response queue...")
        while not audio_playback_queue.empty():
            try:
                audio_playback_queue.get_nowait()
                audio_playback_queue.task_done()
            except asyncio.QueueEmpty:
                break

    # Initialize Gemini Live Engine
    gemini_client = GeminiLive(
        api_key=GEMINI_API_KEY, model=MODEL, input_sample_rate=INPUT_RATE
    )

    # Start concurrent background loops for audio hardware processing
    record_task = asyncio.create_task(mic_audio_recorder(audio_input_queue, input_stream))
    play_task = asyncio.create_task(speaker_audio_player(audio_playback_queue, output_stream))

    print("🎙️  Voice Session Started! Start talking into your microphone... (Ctrl+C to stop)")

    try:
        # Fixed syntax error line: running the asynchronous loop over the session generator directly
        async for event in gemini_client.start_session(
            audio_input_queue=audio_input_queue,
            video_input_queue=video_input_queue,
            text_input_queue=text_input_queue,
            audio_output_callback=audio_output_callback,
            audio_interrupt_callback=audio_interrupt_callback,
        ):
            if event and isinstance(event, dict):
                # Print text transcription alongside the voice output if available
                text_content = event.get("text") or event.get("content")
                if text_content:
                    print(text_content, end="", flush=True)

    except Exception as e:
        import traceback
        print(f"\nSession closed due to error:\n{traceback.format_exc()}")
    finally:
        # Clean up active asynchronous workers
        record_task.cancel()
        play_task.cancel()
        
        # Stop and close hardware stream instances gracefully
        try:
            input_stream.stop_stream()
            input_stream.close()
            output_stream.stop_stream()
            output_stream.close()
            p.terminate()
        except Exception:
            pass
        print("\n🔒 Audio engine safely disconnected.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession stopped by user.")