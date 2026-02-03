"""
AI Music Generation API
Scate AI - AI Engineer Case Study

FastAPI backend for AI music generation and voice cloning.
Integrates ElevenLabs Eleven Music, Kits.AI, and Replicate APIs.
"""

import os
import uuid
import asyncio
import aiohttp
import aiofiles
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Auth imports
# Auth imports
from jose import JWTError, jwt
# from passlib.context import CryptContext
import bcrypt

# ============================================================================
# Configuration
# ============================================================================

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
KITS_API_KEY = os.environ.get("KITS_API_KEY", "")
REPLICATE_API_KEY = os.environ.get("REPLICATE_API_KEY", "")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")

# Auth config
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production-" + str(uuid.uuid4()))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
KITS_BASE_URL = "https://arpeggi.io/api/kits/v1"
REPLICATE_BASE_URL = "https://api.replicate.com/v1"
RUNPOD_BASE_URL = "https://api.runpod.ai/v2"

# Storage paths
UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Password hashing
# Password hashing
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# security = HTTPBearer(auto_error=False)
security = HTTPBearer(auto_error=False)

# ============================================================================
# User Storage (in-memory, replace with database in production)
# ============================================================================

users_store: Dict[str, Dict] = {}  # email -> user data

# ============================================================================
# Auth Helper Functions
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    # bcrypt has a 72 byte limit for passwords
    if len(password.encode('utf-8')) > 72:
        raise HTTPException(status_code=400, detail="Password is too long (max 72 characters)")
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> Optional[str]:
    """Decode JWT token and return user_id if valid"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        return user_id
    except JWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict]:
    """Get current user from JWT token (returns None if not authenticated)"""
    if credentials is None:
        return None
    
    user_id = decode_token(credentials.credentials)
    if user_id is None:
        return None
    
    # Find user by ID
    for email, user in users_store.items():
        if user.get("id") == user_id:
            return user
    
    return None

async def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Require authentication - raises 401 if not authenticated"""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user_id = decode_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    # Find user by ID
    for email, user in users_store.items():
        if user.get("id") == user_id:
            return user
    
    raise HTTPException(status_code=401, detail="User not found")

# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="AI Music Generator",
    description="Generate original music and voice covers using AI",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ============================================================================
# Pydantic Models
# ============================================================================

class PipelineFlow(str, Enum):
    FLOW_A = "original_music"
    FLOW_B = "voice_cover"
    FLOW_C = "combined"

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class MusicGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text description of the music to generate")
    duration_ms: int = Field(default=60000, ge=10000, le=300000, description="Duration in milliseconds")
    instrumental_only: bool = Field(default=False, description="Generate instrumental only")
    lyrics: Optional[str] = Field(default=None, description="Optional lyrics to include")

class CompositionSection(BaseModel):
    name: str
    duration_ms: int
    style: str
    lyrics: Optional[str] = None

class CompositionPlanRequest(BaseModel):
    sections: List[CompositionSection]
    global_style: str

class VoiceCoverRequest(BaseModel):
    voice_model_id: str = Field(..., description="Kits.AI voice model ID")
    conversion_strength: float = Field(default=0.8, ge=0.0, le=1.0)
    pitch_shift: int = Field(default=0, ge=-12, le=12)

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    flow: PipelineFlow
    created_at: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    flow: PipelineFlow
    progress: int
    output_url: Optional[str] = None
    error: Optional[str] = None
    lyrics: Optional[str] = None
    duration_ms: Optional[int] = None

# ============================================================================
# In-Memory Job Store (use Redis in production)
# ============================================================================

jobs_store: Dict[str, Dict[str, Any]] = {}

# ============================================================================
# Custom Models Storage
# ============================================================================

MODELS_DIR = Path("./custom_models")
MODELS_DIR.mkdir(exist_ok=True)

# In-memory index of custom models (persisted to JSON file)
# Structure: { model_id: { user_id, name, model_url, source, created_at } }
custom_models_store: Dict[str, Dict[str, Any]] = {}

def load_custom_models():
    """Load custom models index from disk"""
    global custom_models_store
    index_file = MODELS_DIR / "index.json"
    if index_file.exists():
        import json
        with open(index_file, 'r') as f:
            custom_models_store = json.load(f)
    print(f"[Custom Models] Loaded {len(custom_models_store)} models")

def save_custom_models():
    """Save custom models index to disk"""
    import json
    index_file = MODELS_DIR / "index.json"
    with open(index_file, 'w') as f:
        json.dump(custom_models_store, f, indent=2)

def load_users():
    """Load users from disk"""
    global users_store
    users_file = MODELS_DIR / "users.json"
    if users_file.exists():
        import json
        with open(users_file, 'r') as f:
            users_store = json.load(f)
    print(f"[Users] Loaded {len(users_store)} users")

def save_users():
    """Save users to disk"""
    import json
    users_file = MODELS_DIR / "users.json"
    with open(users_file, 'w') as f:
        json.dump(users_store, f, indent=2)

def get_user_models(user_id: str) -> List[Dict]:
    """Get all models belonging to a specific user"""
    models = []
    for model_id, model_data in custom_models_store.items():
        if model_data.get("user_id") == user_id:
            models.append({
                "id": model_id,
                **model_data
            })
    return models

# Load models and users on startup
load_custom_models()
load_users()

# ============================================================================
# API Clients
# ============================================================================

class ElevenLabsClient:
    """Async client for ElevenLabs Eleven Music API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = ELEVENLABS_BASE_URL
        self.headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
    
    async def compose_music(
        self,
        prompt: str,
        duration_ms: int = 60000,
        instrumental_only: bool = False,
        output_format: str = "mp3_44100_192"
    ) -> bytes:
        """Generate music from text prompt (simple endpoint, no lyrics returned)"""
        async with aiohttp.ClientSession() as session:
            # Note: ElevenLabs API uses 'music_length_ms' not 'duration_ms'
            # and 'force_instrumental' not 'instrumental_only'
            payload = {
                "prompt": prompt,
                "music_length_ms": duration_ms,
                "force_instrumental": instrumental_only,
                "output_format": output_format,
                "model_id": "music_v1"
            }
            
            # Debug logging
            print(f"[ElevenLabs API] Sending request with music_length_ms={duration_ms}, force_instrumental={instrumental_only}")
            print(f"[ElevenLabs API] Full payload: {payload}")
            
            async with session.post(
                f"{self.base_url}/music",
                headers=self.headers,
                json=payload
            ) as response:
                if response.status == 400:
                    error_data = await response.json()
                    raise HTTPException(
                        status_code=400,
                        detail=f"Bad prompt: {error_data.get('message')}. Suggestion: {error_data.get('suggestion')}"
                    )
                
                if response.status != 200:
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"ElevenLabs API error: {await response.text()}"
                    )
                
                return await response.read()
    
    async def compose_music_detailed(
        self,
        prompt: str,
        duration_ms: int = 60000,
        instrumental_only: bool = False,
        output_format: str = "mp3_44100_192"
    ) -> Dict:
        """Generate music and return audio + composition plan with lyrics"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "prompt": prompt,
                "music_length_ms": duration_ms,
                "force_instrumental": instrumental_only,
                "model_id": "music_v1"
            }
            
            print(f"[ElevenLabs API Detailed] Sending request with music_length_ms={duration_ms}")
            
            # Use query parameter for output_format
            url = f"{self.base_url}/music/detailed?output_format={output_format}"
            
            async with session.post(
                url,
                headers=self.headers,
                json=payload
            ) as response:
                if response.status == 400:
                    error_data = await response.json()
                    raise HTTPException(
                        status_code=400,
                        detail=f"Bad prompt: {error_data.get('message')}. Suggestion: {error_data.get('suggestion')}"
                    )
                
                if response.status != 200:
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"ElevenLabs API error: {await response.text()}"
                    )
                
                # Parse multipart response
                content_type = response.headers.get('Content-Type', '')
                
                if 'multipart' in content_type:
                    # Handle multipart response
                    reader = aiohttp.MultipartReader.from_response(response)
                    result = {"audio": None, "composition_plan": None, "song_metadata": None}
                    
                    async for part in reader:
                        if part.headers.get('Content-Type', '').startswith('application/json'):
                            json_data = await part.json()
                            result["composition_plan"] = json_data.get("composition_plan")
                            result["song_metadata"] = json_data.get("song_metadata")
                        else:
                            result["audio"] = await part.read()
                    
                    return result
                else:
                    # Fallback - try to parse as JSON
                    try:
                        data = await response.json()
                        return {
                            "audio": data.get("audio"),
                            "composition_plan": data.get("composition_plan"),
                            "song_metadata": data.get("song_metadata")
                        }
                    except:
                        # Just return audio bytes
                        return {
                            "audio": await response.read(),
                            "composition_plan": None,
                            "song_metadata": None
                        }
    
    async def compose_with_plan(
        self,
        composition_plan: Dict,
        output_format: str = "mp3_44100_192"
    ) -> bytes:
        """Generate music with composition plan"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "composition_plan": composition_plan,
                "output_format": output_format
            }
            
            async with session.post(
                f"{self.base_url}/music/compose",
                headers=self.headers,
                json=payload
            ) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"ElevenLabs API error: {await response.text()}"
                    )
                
                return await response.read()


class KitsAIClient:
    """Async client for Kits.AI API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = KITS_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}"
        }
    
    async def get_voice_models(self, my_models_only: bool = False) -> List[Dict]:
        """Get available voice models"""
        async with aiohttp.ClientSession() as session:
            params = {"myModels": str(my_models_only).lower()}
            
            async with session.get(
                f"{self.base_url}/voice-models",
                headers=self.headers,
                params=params
            ) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Kits.AI API error: {await response.text()}"
                    )
                
                data = await response.json()
                return data.get("data", [])
    
    async def separate_vocals(self, audio_data: bytes) -> Dict[str, str]:
        """Separate vocals from instrumental"""
        async with aiohttp.ClientSession() as session:
            form_data = aiohttp.FormData()
            form_data.add_field('soundFile', audio_data, filename='song.wav')
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # Start separation job
            async with session.post(
                f"{self.base_url}/vocal-separations",
                headers=headers,
                data=form_data
            ) as response:
                if response.status not in [200, 201]:
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Kits.AI API error: {await response.text()}"
                    )
                
                data = await response.json()
                job_id = data.get("id")
            
            # Poll for completion
            result = await self._poll_job(f"/vocal-separations/{job_id}")
            
            return {
                "vocals": result.get("vocalsFileUrl"),
                "instrumental": result.get("instrumentalFileUrl")
            }
    
    async def convert_voice(
        self,
        voice_model_id: str,
        audio_data: bytes,
        conversion_strength: float = 0.8,
        pitch_shift: int = 0
    ) -> str:
        """Convert vocals to different voice"""
        async with aiohttp.ClientSession() as session:
            form_data = aiohttp.FormData()
            form_data.add_field('soundFile', audio_data, filename='vocals.wav')
            form_data.add_field('voiceModelId', str(voice_model_id))
            form_data.add_field('conversionStrength', str(conversion_strength))
            form_data.add_field('pitchShift', str(pitch_shift))
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # Start conversion job
            async with session.post(
                f"{self.base_url}/voice-conversions",
                headers=headers,
                data=form_data
            ) as response:
                if response.status not in [200, 201]:
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Kits.AI API error: {await response.text()}"
                    )
                
                data = await response.json()
                job_id = data.get("id")
            
            # Poll for completion
            result = await self._poll_job(f"/voice-conversions/{job_id}")
            
            return result.get("outputFileUrl")
    
    async def _poll_job(self, endpoint: str, max_attempts: int = 120, interval: int = 2) -> Dict:
        """Poll job endpoint until completion"""
        async with aiohttp.ClientSession() as session:
            for _ in range(max_attempts):
                async with session.get(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers
                ) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Kits.AI API error: {await response.text()}"
                        )
                    
                    data = await response.json()
                    status = data.get("status")
                    
                    if status == "completed":
                        return data
                    elif status == "failed":
                        raise HTTPException(
                            status_code=500,
                            detail=f"Job failed: {data.get('error')}"
                        )
                    
                    await asyncio.sleep(interval)
            
            raise HTTPException(status_code=408, detail="Job timed out")


class ReplicateClient:
    """Async client for Replicate API - Voice Cloning"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = REPLICATE_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def clone_voice_and_convert(
        self,
        voice_sample_url: str,
        source_audio_url: str,
        pitch_shift: int = 0
    ) -> str:
        """
        Clone a voice from a sample and convert source audio to that voice.
        Uses zsxkib/realistic-voice-cloning model.
        
        Returns: URL to the converted audio
        """
        async with aiohttp.ClientSession() as session:
            # Use the realistic-voice-cloning model
            payload = {
                "version": "0a9c7c558af4c0f20667c1bd1260ce32a2879944a0b9e44e1398660c077b1550",
                "input": {
                    "song_input": source_audio_url,
                    "rvc_model": "CUSTOM",
                    "custom_rvc_model_download_url": voice_sample_url,
                    "pitch_change": pitch_shift,
                    "index_rate": 0.5,
                    "filter_radius": 3,
                    "rms_mix_rate": 0.25,
                    "protect": 0.33,
                    "main_vocals_volume_change": 0,
                    "backup_vocals_volume_change": 0,
                    "instrumental_volume_change": 0,
                    "reverb_size": 0.15,
                    "reverb_wetness": 0.2,
                    "reverb_dryness": 0.8,
                    "reverb_damping": 0.7
                }
            }
            
            print(f"[Replicate] Starting voice cloning prediction...")
            
            # Create prediction
            async with session.post(
                f"{self.base_url}/predictions",
                headers=self.headers,
                json=payload
            ) as response:
                if response.status not in [200, 201]:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Replicate API error: {error_text}"
                    )
                
                prediction = await response.json()
                prediction_id = prediction.get("id")
                print(f"[Replicate] Created prediction: {prediction_id}")
            
            # Poll for completion
            result = await self._poll_prediction(prediction_id)
            
            # Get output URL
            output = result.get("output")
            if isinstance(output, str):
                return output
            elif isinstance(output, dict):
                return output.get("audio") or output.get("output")
            elif isinstance(output, list) and len(output) > 0:
                return output[0]
            else:
                raise HTTPException(status_code=500, detail="No output from voice cloning")
    
    async def voice_to_voice_simple(
        self,
        voice_sample_data: bytes,
        source_audio_data: bytes,
        pitch_shift: int = 0
    ) -> bytes:
        """
        Simplified voice-to-voice conversion using OpenVoice or similar.
        Takes raw audio bytes and returns converted audio bytes.
        """
        async with aiohttp.ClientSession() as session:
            # First, we need to upload the files or use data URIs
            # For Replicate, we'll use base64 encoded data URIs
            voice_b64 = base64.b64encode(voice_sample_data).decode('utf-8')
            source_b64 = base64.b64encode(source_audio_data).decode('utf-8')
            
            voice_uri = f"data:audio/wav;base64,{voice_b64}"
            source_uri = f"data:audio/wav;base64,{source_b64}"
            
            # Use openvoice model for instant cloning
            payload = {
                "version": "c14768ec4d2a0a30b06d64758ab1dac8c9d3e3cf97c24d37e6f4e93f2f4d83e1",
                "input": {
                    "audio": source_uri,
                    "reference_speaker": voice_uri,
                    "speed": 1.0
                }
            }
            
            print(f"[Replicate] Starting OpenVoice conversion...")
            
            async with session.post(
                f"{self.base_url}/predictions",
                headers=self.headers,
                json=payload
            ) as response:
                if response.status not in [200, 201]:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Replicate API error: {error_text}"
                    )
                
                prediction = await response.json()
                prediction_id = prediction.get("id")
            
            # Poll for completion
            result = await self._poll_prediction(prediction_id)
            output_url = result.get("output")
            
            # Download the result
            async with session.get(output_url) as resp:
                return await resp.read()
    
    async def _poll_prediction(self, prediction_id: str, max_attempts: int = 180, interval: int = 2) -> Dict:
        """Poll prediction until completion"""
        async with aiohttp.ClientSession() as session:
            for attempt in range(max_attempts):
                async with session.get(
                    f"{self.base_url}/predictions/{prediction_id}",
                    headers=self.headers
                ) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Replicate API error: {await response.text()}"
                        )
                    
                    data = await response.json()
                    status = data.get("status")
                    
                    if status == "succeeded":
                        print(f"[Replicate] Prediction completed successfully")
                        return data
                    elif status == "failed":
                        error = data.get("error", "Unknown error")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Replicate prediction failed: {error}"
                        )
                    elif status == "canceled":
                        raise HTTPException(
                            status_code=500,
                            detail="Replicate prediction was canceled"
                        )
                    
                    if attempt % 10 == 0:
                        print(f"[Replicate] Waiting... status={status}, attempt={attempt}")
                    
                    await asyncio.sleep(interval)
            
            raise HTTPException(status_code=408, detail="Replicate prediction timed out")


# Initialize clients
elevenlabs_client = ElevenLabsClient(ELEVENLABS_API_KEY)
kits_client = KitsAIClient(KITS_API_KEY)
replicate_client = ReplicateClient(REPLICATE_API_KEY)

# ============================================================================
# Background Tasks
# ============================================================================

async def process_music_generation(job_id: str, request: Dict):
    """Background task for music generation (Flow A)"""
    try:
        jobs_store[job_id]["status"] = JobStatus.PROCESSING
        jobs_store[job_id]["progress"] = 10
        
        # Store the requested duration
        duration_ms = request.get("duration_ms", 60000)
        jobs_store[job_id]["duration_ms"] = duration_ms
        instrumental_only = request.get("instrumental_only", False)
        
        # Build prompt
        prompt = request["prompt"]
        user_lyrics = request.get("lyrics")
        
        if user_lyrics:
            prompt += f"\n\nLyrics:\n{user_lyrics}"
            jobs_store[job_id]["lyrics"] = user_lyrics
        
        jobs_store[job_id]["progress"] = 30
        
        # Use detailed endpoint to get lyrics back (only if not instrumental and no user lyrics)
        if not instrumental_only and not user_lyrics:
            try:
                result = await elevenlabs_client.compose_music_detailed(
                    prompt=prompt,
                    duration_ms=duration_ms,
                    instrumental_only=instrumental_only
                )
                
                audio_data = result.get("audio")
                composition_plan = result.get("composition_plan")
                
                # Extract lyrics from composition plan
                if composition_plan and composition_plan.get("sections"):
                    lyrics_lines = []
                    for section in composition_plan["sections"]:
                        section_name = section.get("section_name", "")
                        lines = section.get("lines", [])
                        if lines:
                            lyrics_lines.append(f"[{section_name}]")
                            lyrics_lines.extend(lines)
                            lyrics_lines.append("")
                    
                    if lyrics_lines:
                        generated_lyrics = "\n".join(lyrics_lines)
                        jobs_store[job_id]["lyrics"] = generated_lyrics
                        print(f"[Music Generation] Extracted lyrics:\n{generated_lyrics[:200]}...")
                
            except Exception as e:
                print(f"[Music Generation] Detailed endpoint failed, falling back to simple: {e}")
                # Fallback to simple endpoint
                audio_data = await elevenlabs_client.compose_music(
                    prompt=prompt,
                    duration_ms=duration_ms,
                    instrumental_only=instrumental_only
                )
        else:
            # Use simple endpoint for instrumental or when user provided lyrics
            audio_data = await elevenlabs_client.compose_music(
                prompt=prompt,
                duration_ms=duration_ms,
                instrumental_only=instrumental_only
            )
        
        jobs_store[job_id]["progress"] = 80
        
        # Save output
        output_filename = f"{job_id}.mp3"
        output_path = OUTPUT_DIR / output_filename
        
        async with aiofiles.open(output_path, 'wb') as f:
            await f.write(audio_data)
        
        jobs_store[job_id]["status"] = JobStatus.COMPLETED
        jobs_store[job_id]["progress"] = 100
        jobs_store[job_id]["output_url"] = f"/outputs/{output_filename}"
        
    except Exception as e:
        jobs_store[job_id]["status"] = JobStatus.FAILED
        jobs_store[job_id]["error"] = str(e)



async def process_voice_cover(job_id: str, song_path: str, voice_sample_path: Optional[str], voice_model_id: Optional[str], request: Dict):
    """Background task for voice cover generation (Flow B)
    
    Voice sources:
    1. 'url' - Use direct model URL (Replicate downloads it)
    2. 'custom' or 'upload_rvc' - Use RVC model with Replicate (requires upload)
    3. 'library' - Use Kits.AI voice library
    """
    try:
        jobs_store[job_id]["status"] = JobStatus.PROCESSING
        jobs_store[job_id]["progress"] = 5
        
        pitch_shift = request.get("pitch_shift", 0)
        voice_source = request.get("voice_source", "library")
        rvc_model_path = request.get("rvc_model_path")
        rvc_index_path = request.get("rvc_index_path")
        direct_model_url = request.get("direct_model_url")
        
        # ========================================
        # Option 1: Direct URL (easiest - no upload needed!)
        # This handles both 'url' voice_source AND 'custom' models with model_url
        # ========================================
        if direct_model_url:
            print(f"[Voice Cover] Using direct model URL: {direct_model_url}")
            
            # Read song file
            async with aiofiles.open(song_path, 'rb') as f:
                song_data = await f.read()
            
            jobs_store[job_id]["progress"] = 20
            
            # Convert song to base64
            song_b64 = base64.b64encode(song_data).decode('utf-8')
            
            jobs_store[job_id]["progress"] = 30
            
            async with aiohttp.ClientSession() as session:
                # Build the conversion payload - Replicate will download the model directly!
                inference_payload = {
                    "version": "0a9c7c558af4c0f20667c1bd1260ce32a2879944a0b9e44e1398660c077b1550",
                    "input": {
                        "protect": 0.33,
                        "rvc_model": "CUSTOM",
                        "custom_rvc_model_download_url": direct_model_url,
                        "index_rate": 0.5,
                        "song_input": f"data:audio/mp3;base64,{song_b64}",
                        "reverb_size": 0.15,
                        "pitch_change": "no-change",
                        "rms_mix_rate": 0.25,
                        "filter_radius": 3,
                        "reverb_damping": 0.7,
                        "reverb_dryness": 0.8,
                        "reverb_wetness": 0.2,
                        "output_format": "mp3",
                        "crepe_hop_length": 128,
                        "main_vocals_volume_change": 0,
                        "pitch_change_all": pitch_shift,
                        "instrumental_volume_change": 0,
                        "backup_vocals_volume_change": 0
                    }
                }
                
                headers = {
                    "Authorization": f"Bearer {REPLICATE_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                print(f"[Voice Cover] ✅ Using CUSTOM model from URL: {direct_model_url}")
                print(f"[Voice Cover] Starting voice conversion with Replicate...")
                
                async with session.post(
                    f"{REPLICATE_BASE_URL}/predictions",
                    headers=headers,
                    json=inference_payload
                ) as response:
                    if response.status not in [200, 201]:
                        error_text = await response.text()
                        raise Exception(f"Replicate API error: {error_text}")
                    
                    prediction = await response.json()
                    prediction_id = prediction.get("id")
                    print(f"[Voice Cover] Conversion started: {prediction_id}")
                
                jobs_store[job_id]["progress"] = 40
                
                # Poll for conversion completion
                for attempt in range(300):  # 10 minutes timeout
                    async with session.get(
                        f"{REPLICATE_BASE_URL}/predictions/{prediction_id}",
                        headers=headers
                    ) as response:
                        data = await response.json()
                        status = data.get("status")
                        
                        if status == "succeeded":
                            output_url = data.get("output")
                            print(f"[Voice Cover] ✅ Conversion completed!")
                            break
                        elif status in ["failed", "canceled"]:
                            error_msg = data.get('error', 'Unknown error')
                            logs = data.get('logs', '')
                            print(f"[Voice Cover] ❌ Conversion {status}: {error_msg}")
                            if logs:
                                print(f"[Voice Cover] Logs: {logs[-1000:]}")
                            raise Exception(f"Conversion {status}: {error_msg}")
                        
                        if attempt % 15 == 0:
                            progress = min(40 + (attempt // 5), 95)
                            jobs_store[job_id]["progress"] = progress
                            print(f"[Voice Cover] Status: {status}, attempt: {attempt}")
                        
                        await asyncio.sleep(2)
                else:
                    raise Exception("Conversion timed out after 10 minutes")
                
                jobs_store[job_id]["progress"] = 95
                
                # Download result
                async with session.get(output_url) as resp:
                    converted_data = await resp.read()
        
        # ========================================
        # Option 2: Local RVC model file (requires upload)
        # ========================================
        elif voice_source in ['custom', 'upload_rvc'] and rvc_model_path:
            # ========================================
            # Use RVC model with Replicate
            # ========================================
            print(f"[Voice Cover] Using RVC model: {rvc_model_path}")
            
            # Read song file
            async with aiofiles.open(song_path, 'rb') as f:
                song_data = await f.read()
            
            jobs_store[job_id]["progress"] = 15
            
            # Read the RVC model file
            async with aiofiles.open(rvc_model_path, 'rb') as f:
                model_data = await f.read()
            
            # Read index file if available
            index_data = None
            if rvc_index_path:
                try:
                    async with aiofiles.open(rvc_index_path, 'rb') as f:
                        index_data = await f.read()
                    print(f"[Voice Cover] Index file loaded: {len(index_data)} bytes")
                except Exception as e:
                    print(f"[Voice Cover] No index file: {e}")
            
            jobs_store[job_id]["progress"] = 25
            
            # Create a ZIP file containing the model (required format for Replicate)
            import zipfile
            import io
            
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                # The model name inside ZIP should match what we tell Replicate
                zf.writestr("model.pth", model_data)
                if index_data:
                    zf.writestr("model.index", index_data)
            
            zip_buffer.seek(0)
            zip_data = zip_buffer.read()
            
            print(f"[Voice Cover] Created ZIP file: {len(zip_data)} bytes (model: {len(model_data)} bytes)")
            
            jobs_store[job_id]["progress"] = 35
            
            # Upload to Replicate's Files API (most reliable method)
            model_url = None
            async with aiohttp.ClientSession() as session:
                print(f"[Voice Cover] Uploading model to Replicate Files API...")
                
                # Use Replicate's native file upload API
                form_data = aiohttp.FormData()
                form_data.add_field(
                    'content', 
                    zip_data, 
                    filename='rvc_model.zip',
                    content_type='application/zip'
                )
                
                try:
                    async with session.post(
                        'https://api.replicate.com/v1/files',
                        headers={'Authorization': f'Bearer {REPLICATE_API_KEY}'},
                        data=form_data,
                        timeout=aiohttp.ClientTimeout(total=300)
                    ) as resp:
                        resp_text = await resp.text()
                        print(f"[Voice Cover] Replicate Files API response status: {resp.status}")
                        print(f"[Voice Cover] Replicate Files API response: {resp_text[:500]}")
                        
                        if resp.status == 200 or resp.status == 201:
                            result = await resp.json() if resp.content_type == 'application/json' else {}
                            # The URL is in urls.get
                            model_url = result.get('urls', {}).get('get')
                            if model_url:
                                print(f"[Voice Cover] Model uploaded to Replicate: {model_url}")
                            else:
                                print(f"[Voice Cover] No URL in response: {result}")
                        else:
                            print(f"[Voice Cover] Replicate upload failed: {resp_text}")
                except Exception as e:
                    print(f"[Voice Cover] Replicate Files API error: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Fallback: Try using HuggingFace-style direct URL if available
                # If the model was downloaded from voice-models.com, it might have a direct URL
                if not model_url:
                    print(f"[Voice Cover] Trying fallback upload methods...")
                    
                    # Try 0x0.st
                    try:
                        form_data = aiohttp.FormData()
                        form_data.add_field('file', zip_data, filename='rvc_model.zip', content_type='application/zip')
                        async with session.post('https://0x0.st', data=form_data, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                            if resp.status == 200:
                                model_url = (await resp.text()).strip()
                                print(f"[Voice Cover] Uploaded to 0x0.st: {model_url}")
                    except Exception as e:
                        print(f"[Voice Cover] 0x0.st failed: {e}")
                
                jobs_store[job_id]["progress"] = 45
                
                if not model_url:
                    error_msg = "Failed to upload model to any file hosting service. The voice cover will use default Squidward voice."
                    print(f"[Voice Cover] ERROR: {error_msg}")
                    jobs_store[job_id]["error"] = error_msg
                
                # Convert song to base64
                song_b64 = base64.b64encode(song_data).decode('utf-8')
                
                # Build the conversion payload
                inference_payload = {
                    "version": "0a9c7c558af4c0f20667c1bd1260ce32a2879944a0b9e44e1398660c077b1550",
                    "input": {
                        "protect": 0.33,
                        "index_rate": 0.5 if index_data else 0,
                        "song_input": f"data:audio/mp3;base64,{song_b64}",
                        "reverb_size": 0.15,
                        "pitch_change": "no-change",
                        "rms_mix_rate": 0.25,
                        "filter_radius": 3,
                        "reverb_damping": 0.7,
                        "reverb_dryness": 0.8,
                        "reverb_wetness": 0.2,
                        "output_format": "mp3",
                        "crepe_hop_length": 128,
                        "main_vocals_volume_change": 0,
                        "pitch_change_all": pitch_shift,
                        "instrumental_volume_change": 0,
                        "backup_vocals_volume_change": 0
                    }
                }
                
                # Use custom model if upload succeeded
                if model_url:
                    inference_payload["input"]["rvc_model"] = "CUSTOM"
                    inference_payload["input"]["custom_rvc_model_download_url"] = model_url
                    print(f"[Voice Cover] ✅ Using CUSTOM RVC model from: {model_url}")
                else:
                    inference_payload["input"]["rvc_model"] = "Squidward"
                    print(f"[Voice Cover] ⚠️ Falling back to Squidward model (upload failed)")
                
                headers = {
                    "Authorization": f"Bearer {REPLICATE_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                print(f"[Voice Cover] Starting voice conversion...")
                print(f"[Voice Cover] rvc_model = {inference_payload['input']['rvc_model']}")
                if 'custom_rvc_model_download_url' in inference_payload['input']:
                    print(f"[Voice Cover] custom_rvc_model_download_url = {inference_payload['input']['custom_rvc_model_download_url']}")
                
                async with session.post(
                    f"{REPLICATE_BASE_URL}/predictions",
                    headers=headers,
                    json=inference_payload
                ) as response:
                    if response.status not in [200, 201]:
                        error_text = await response.text()
                        raise Exception(f"Replicate API error: {error_text}")
                    
                    prediction = await response.json()
                    prediction_id = prediction.get("id")
                    print(f"[Voice Cover] Conversion started: {prediction_id}")
                
                jobs_store[job_id]["progress"] = 55
                
                # Poll for conversion completion
                for attempt in range(180):
                    async with session.get(
                        f"{REPLICATE_BASE_URL}/predictions/{prediction_id}",
                        headers=headers
                    ) as response:
                        data = await response.json()
                        status = data.get("status")
                        
                        if status == "succeeded":
                            output_url = data.get("output")
                            print(f"[Voice Cover] ✅ Conversion completed!")
                            break
                        elif status in ["failed", "canceled"]:
                            error_msg = data.get('error', 'Unknown error')
                            logs = data.get('logs', '')
                            print(f"[Voice Cover] ❌ Conversion {status}: {error_msg}")
                            print(f"[Voice Cover] Logs: {logs[-500:] if logs else 'No logs'}")
                            raise Exception(f"Conversion {status}: {error_msg}")
                        
                        if attempt % 10 == 0:
                            progress = min(55 + (attempt // 4), 95)
                            jobs_store[job_id]["progress"] = progress
                            print(f"[Voice Cover] Status: {status}, attempt: {attempt}")
                        
                        await asyncio.sleep(2)
                else:
                    raise Exception("Conversion timed out after 6 minutes")
                
                jobs_store[job_id]["progress"] = 95
                
                # Download result
                async with session.get(output_url) as resp:
                    converted_data = await resp.read()
        
        elif voice_model_id:
            # ========================================
            # Use Kits.AI for voice conversion
            # ========================================
            print(f"[Voice Cover] Using Kits.AI voice model: {voice_model_id}")
            
            async with aiofiles.open(song_path, 'rb') as f:
                song_data = await f.read()
            
            jobs_store[job_id]["progress"] = 20
            
            # Separate vocals
            print(f"[Voice Cover] Separating vocals...")
            separation = await kits_client.separate_vocals(song_data)
            
            jobs_store[job_id]["progress"] = 50
            
            # Download vocals and instrumental
            async with aiohttp.ClientSession() as session:
                async with session.get(separation["vocals"]) as resp:
                    vocals_data = await resp.read()
                async with session.get(separation["instrumental"]) as resp:
                    instrumental_data = await resp.read()
            
            jobs_store[job_id]["progress"] = 60
            
            # Convert vocals
            print(f"[Voice Cover] Converting vocals...")
            converted_url = await kits_client.convert_voice(
                voice_model_id=voice_model_id,
                audio_data=vocals_data,
                conversion_strength=0.8,
                pitch_shift=pitch_shift
            )
            
            jobs_store[job_id]["progress"] = 80
            
            # Download converted
            async with aiohttp.ClientSession() as session:
                async with session.get(converted_url) as resp:
                    converted_data = await resp.read()
            
            # Save instrumental
            instrumental_filename = f"{job_id}_instrumental.mp3"
            instrumental_path = OUTPUT_DIR / instrumental_filename
            async with aiofiles.open(instrumental_path, 'wb') as f:
                await f.write(instrumental_data)
            jobs_store[job_id]["instrumental_url"] = f"/outputs/{instrumental_filename}"
        
        else:
            raise Exception("No voice source specified")
        
        # Save final output
        output_filename = f"{job_id}.mp3"
        output_path = OUTPUT_DIR / output_filename
        
        async with aiofiles.open(output_path, 'wb') as f:
            await f.write(converted_data)
        
        jobs_store[job_id]["status"] = JobStatus.COMPLETED
        jobs_store[job_id]["progress"] = 100
        jobs_store[job_id]["output_url"] = f"/outputs/{output_filename}"
        print(f"[Voice Cover] Completed! Output: {output_filename}")
        
    except Exception as e:
        print(f"[Voice Cover] Error: {e}")
        import traceback
        traceback.print_exc()
        jobs_store[job_id]["status"] = JobStatus.FAILED
        jobs_store[job_id]["error"] = str(e)

async def process_combined_generation(job_id: str, model_url: str, request: Dict):
    """Background task for combined generation (Flow C)
    
    Uses saved custom model URL for voice conversion via Replicate.
    """
    try:
        jobs_store[job_id]["status"] = JobStatus.PROCESSING
        jobs_store[job_id]["progress"] = 5
        
        # Step 1: Generate original music
        prompt = request["prompt"]
        if request.get("lyrics"):
            prompt += f"\n\nLyrics:\n{request['lyrics']}"
        
        print(f"[Combined] Step 1: Generating music with prompt: {prompt[:100]}...")
        jobs_store[job_id]["progress"] = 10
        
        audio_data = await elevenlabs_client.compose_music(
            prompt=prompt,
            duration_ms=request.get("duration_ms", 60000),
            instrumental_only=False  # Need vocals for conversion
        )
        
        print(f"[Combined] Music generated: {len(audio_data)} bytes")
        jobs_store[job_id]["progress"] = 40
        
        # Step 2: Save generated music temporarily
        temp_song_path = UPLOAD_DIR / f"{job_id}_generated.mp3"
        async with aiofiles.open(temp_song_path, 'wb') as f:
            await f.write(audio_data)
        
        # Step 3: Use Replicate for voice conversion (same as Flow B)
        print(f"[Combined] Step 2: Converting voice with model: {model_url}")
        
        # Convert to base64
        song_b64 = base64.b64encode(audio_data).decode('utf-8')
        
        jobs_store[job_id]["progress"] = 50
        
        async with aiohttp.ClientSession() as session:
            # Build the conversion payload
            inference_payload = {
                "version": "0a9c7c558af4c0f20667c1bd1260ce32a2879944a0b9e44e1398660c077b1550",
                "input": {
                    "protect": 0.33,
                    "rvc_model": "CUSTOM",
                    "custom_rvc_model_download_url": model_url,
                    "index_rate": 0.5,
                    "song_input": f"data:audio/mp3;base64,{song_b64}",
                    "reverb_size": 0.15,
                    "pitch_change": "no-change",
                    "rms_mix_rate": 0.25,
                    "filter_radius": 3,
                    "reverb_damping": 0.7,
                    "reverb_dryness": 0.8,
                    "reverb_wetness": 0.2,
                    "output_format": "mp3",
                    "crepe_hop_length": 128,
                    "main_vocals_volume_change": 0,
                    "pitch_change_all": 0,
                    "instrumental_volume_change": 0,
                    "backup_vocals_volume_change": 0
                }
            }
            
            headers = {
                "Authorization": f"Bearer {REPLICATE_API_KEY}",
                "Content-Type": "application/json"
            }
            
            print(f"[Combined] Starting Replicate voice conversion...")
            
            async with session.post(
                f"{REPLICATE_BASE_URL}/predictions",
                headers=headers,
                json=inference_payload
            ) as response:
                if response.status not in [200, 201]:
                    error_text = await response.text()
                    raise Exception(f"Replicate API error: {error_text}")
                
                prediction = await response.json()
                prediction_id = prediction.get("id")
                print(f"[Combined] Conversion started: {prediction_id}")
            
            jobs_store[job_id]["progress"] = 60
            
            # Poll for conversion completion
            for attempt in range(300):  # 10 minutes timeout
                async with session.get(
                    f"{REPLICATE_BASE_URL}/predictions/{prediction_id}",
                    headers=headers
                ) as response:
                    data = await response.json()
                    status = data.get("status")
                    
                    if status == "succeeded":
                        output_url = data.get("output")
                        print(f"[Combined] ✅ Voice conversion completed!")
                        break
                    elif status in ["failed", "canceled"]:
                        error_msg = data.get('error', 'Unknown error')
                        logs = data.get('logs', '')
                        print(f"[Combined] ❌ Conversion {status}: {error_msg}")
                        if logs:
                            print(f"[Combined] Logs: {logs[-500:]}")
                        raise Exception(f"Voice conversion {status}: {error_msg}")
                    
                    if attempt % 15 == 0:
                        progress = min(60 + (attempt // 5), 90)
                        jobs_store[job_id]["progress"] = progress
                        print(f"[Combined] Status: {status}, attempt: {attempt}")
                    
                    await asyncio.sleep(2)
            else:
                raise Exception("Voice conversion timed out after 10 minutes")
            
            jobs_store[job_id]["progress"] = 95
            
            # Download result
            async with session.get(output_url) as resp:
                final_audio = await resp.read()
        
        # Save final output
        output_filename = f"{job_id}.mp3"
        output_path = OUTPUT_DIR / output_filename
        
        async with aiofiles.open(output_path, 'wb') as f:
            await f.write(final_audio)
        
        # Clean up temp file
        try:
            temp_song_path.unlink()
        except:
            pass
        
        jobs_store[job_id]["status"] = JobStatus.COMPLETED
        jobs_store[job_id]["progress"] = 100
        jobs_store[job_id]["output_url"] = f"/outputs/{output_filename}"
        print(f"[Combined] ✅ Completed! Output: {output_filename}")
        
    except Exception as e:
        print(f"[Combined] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        jobs_store[job_id]["status"] = JobStatus.FAILED
        jobs_store[job_id]["error"] = str(e)

# ============================================================================
# Auth Models
# ============================================================================

class UserSignup(BaseModel):
    email: str
    password: str
    name: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str]
    created_at: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

# ============================================================================
# Auth Endpoints
# ============================================================================

@app.post("/api/auth/signup", response_model=AuthResponse)
async def signup(user_data: UserSignup):
    """Create a new user account"""
    email = user_data.email.lower().strip()
    
    # Validate email format
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Invalid email format")
    
    # Check if email already exists
    if email in users_store:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Validate password
    if len(user_data.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    
    # Create user
    try:
        user_id = str(uuid.uuid4())
        print(f"[Auth] Hashing password for {email}")
        hashed_password = get_password_hash(user_data.password)
        
        user = {
            "id": user_id,
            "email": email,
            "name": user_data.name or email.split("@")[0],
            "password_hash": hashed_password,
            "created_at": datetime.utcnow().isoformat()
        }
        
        users_store[email] = user
        print(f"[Auth] Saving user {email} to disk")
        save_users()
        print(f"[Auth] User saved successfully")
        
        # Create access token
        access_token = create_access_token(data={"sub": user_id})
        
        return AuthResponse(
            access_token=access_token,
            user=UserResponse(
                id=user_id,
                email=email,
                name=user["name"],
                created_at=user["created_at"]
            )
        )
    except Exception as e:
        print(f"[Auth] Signup failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")

@app.post("/api/auth/login", response_model=AuthResponse)
async def login(user_data: UserLogin):
    """Login with email and password"""
    email = user_data.email.lower().strip()
    
    # Check if user exists
    if email not in users_store:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    user = users_store[email]
    
    # Verify password
    if not verify_password(user_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Create access token
    access_token = create_access_token(data={"sub": user["id"]})
    
    return AuthResponse(
        access_token=access_token,
        user=UserResponse(
            id=user["id"],
            email=email,
            name=user["name"],
            created_at=user["created_at"]
        )
    )

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: Dict = Depends(require_auth)):
    """Get current user info"""
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        name=current_user["name"],
        created_at=current_user["created_at"]
    )

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "AI Music Generator API",
        "version": "1.0.0",
        "endpoints": {
            "auth_signup": "/api/auth/signup",
            "auth_login": "/api/auth/login",
            "auth_me": "/api/auth/me",
            "generate_music": "/api/generate/music",
            "generate_cover": "/api/generate/cover",
            "generate_combined": "/api/generate/combined",
            "job_status": "/api/jobs/{job_id}",
            "voice_models": "/api/voice-models"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "elevenlabs_configured": bool(ELEVENLABS_API_KEY),
        "kits_configured": bool(KITS_API_KEY)
    }


# ============================================================================
# Flow A: Original Music Generation
# ============================================================================

@app.post("/api/generate/music", response_model=JobResponse)
async def generate_music(
    request: MusicGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(require_auth)
):
    """
    Generate original music from text prompt (Flow A)
    
    Uses ElevenLabs Eleven Music API to create original songs.
    """
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ElevenLabs API key not configured")
    
    user_id = current_user["id"]
    job_id = str(uuid.uuid4())
    
    # Log the request for debugging
    print(f"[Music Generation] Job {job_id} for user {user_id}: duration_ms={request.duration_ms}, instrumental={request.instrumental_only}")
    
    jobs_store[job_id] = {
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "flow": PipelineFlow.FLOW_A,
        "created_at": datetime.utcnow().isoformat(),
        "progress": 0,
        "output_url": None,
        "error": None,
        "lyrics": request.lyrics,
        "duration_ms": request.duration_ms
    }
    
    background_tasks.add_task(
        process_music_generation,
        job_id,
        request.dict()
    )
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        flow=PipelineFlow.FLOW_A,
        created_at=jobs_store[job_id]["created_at"],
        message=f"Music generation job started (duration: {request.duration_ms}ms)"
    )


@app.post("/api/generate/music/composition", response_model=JobResponse)
async def generate_music_with_plan(
    request: CompositionPlanRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(require_auth)
):
    """
    Generate music with detailed composition plan (Flow A variant)
    
    Allows section-by-section control over the generated song.
    """
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ElevenLabs API key not configured")
    
    user_id = current_user["id"]
    job_id = str(uuid.uuid4())
    
    jobs_store[job_id] = {
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "flow": PipelineFlow.FLOW_A,
        "created_at": datetime.utcnow().isoformat(),
        "progress": 0,
        "output_url": None,
        "error": None
    }
    
    # Convert to composition plan format
    composition_plan = {
        "sections": [s.dict() for s in request.sections],
        "global_style": request.global_style
    }
    
    async def process_composition(job_id: str, plan: Dict):
        try:
            jobs_store[job_id]["status"] = JobStatus.PROCESSING
            jobs_store[job_id]["progress"] = 20
            
            audio_data = await elevenlabs_client.compose_with_plan(plan)
            
            jobs_store[job_id]["progress"] = 80
            
            output_filename = f"{job_id}.mp3"
            output_path = OUTPUT_DIR / output_filename
            
            async with aiofiles.open(output_path, 'wb') as f:
                await f.write(audio_data)
            
            jobs_store[job_id]["status"] = JobStatus.COMPLETED
            jobs_store[job_id]["progress"] = 100
            jobs_store[job_id]["output_url"] = f"/outputs/{output_filename}"
            
        except Exception as e:
            jobs_store[job_id]["status"] = JobStatus.FAILED
            jobs_store[job_id]["error"] = str(e)
    
    background_tasks.add_task(process_composition, job_id, composition_plan)
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        flow=PipelineFlow.FLOW_A,
        created_at=jobs_store[job_id]["created_at"],
        message="Composition generation job started"
    )


# ============================================================================
# Flow B: Voice/Cover Generation
# ============================================================================

@app.post("/api/generate/cover", response_model=JobResponse)
async def generate_cover(
    background_tasks: BackgroundTasks,
    pitch_shift: int = Form(0),
    song_file: UploadFile = File(...),
    voice_source: str = Form("library"),
    # For direct URL (recommended - no upload needed!)
    model_url: Optional[str] = Form(None),
    # For custom models
    custom_model_id: Optional[str] = Form(None),
    # For uploaded RVC models (one-time use)
    rvc_model_file: Optional[UploadFile] = File(None),
    rvc_index_file: Optional[UploadFile] = File(None),
    # For Kits.AI library
    voice_model_id: Optional[str] = Form(None),
    current_user: Dict = Depends(require_auth)
):
    """
    Generate cover with voice conversion (Flow B)
    
    Voice sources:
    1. 'url' - Use a direct URL to an RVC model (from voice-models.com, HuggingFace, etc.)
    2. 'custom' - Use a saved custom model (custom_model_id)
    3. 'upload_rvc' - Upload RVC model files directly (rvc_model_file, rvc_index_file)
    4. 'library' - Use Kits.AI voice library (voice_model_id)
    """
    user_id = current_user["id"]
    job_id = str(uuid.uuid4())
    
    # Validate based on voice source
    if voice_source == 'url':
        if not model_url:
            raise HTTPException(status_code=400, detail="model_url is required for URL voice source")
        if not REPLICATE_API_KEY:
            raise HTTPException(status_code=500, detail="Replicate API key not configured")
        print(f"[Cover Generation] Using direct model URL: {model_url}")
    elif voice_source == 'custom':
        if not custom_model_id:
            raise HTTPException(status_code=400, detail="custom_model_id is required for custom voice source")
        
        # Check both default models and custom models
        found_model = False
        for default_model in DEFAULT_VOICE_MODELS:
            if default_model["id"] == custom_model_id:
                found_model = True
                break
        if not found_model and custom_model_id not in custom_models_store:
            raise HTTPException(status_code=404, detail="Voice model not found")
        
        if not REPLICATE_API_KEY:
            raise HTTPException(status_code=500, detail="Replicate API key not configured")
    elif voice_source == 'upload_rvc':
        if not rvc_model_file:
            raise HTTPException(status_code=400, detail="rvc_model_file is required for upload_rvc voice source")
        if not REPLICATE_API_KEY:
            raise HTTPException(status_code=500, detail="Replicate API key not configured")
    elif voice_source == 'library':
        if not voice_model_id:
            raise HTTPException(status_code=400, detail="voice_model_id is required for library voice source")
        if not KITS_API_KEY:
            raise HTTPException(status_code=500, detail="Kits.AI API key not configured")
    else:
        raise HTTPException(status_code=400, detail=f"Invalid voice_source: {voice_source}")
    
    # Save uploaded song
    song_path = UPLOAD_DIR / f"{job_id}_song.mp3"
    async with aiofiles.open(song_path, 'wb') as f:
        content = await song_file.read()
        await f.write(content)
    
    # Handle different voice sources
    rvc_model_path = None
    rvc_index_path = None
    direct_model_url = None
    
    if voice_source == 'url':
        # Use the URL directly - no need to upload!
        direct_model_url = model_url
        print(f"[Cover Generation] Will use direct URL: {direct_model_url}")
        
    elif voice_source == 'custom':
        # Look up model from defaults or custom store
        model_data = None
        for default_model in DEFAULT_VOICE_MODELS:
            if default_model["id"] == custom_model_id:
                model_data = default_model
                break
        if not model_data:
            model_data = custom_models_store[custom_model_id]
        
        # Check if it's a URL-based model or a file-based model
        if model_data.get("model_url"):
            direct_model_url = model_data.get("model_url")
            print(f"[Cover Generation] Using custom URL model: {model_data.get('name')} -> {direct_model_url}")
        else:
            rvc_model_path = model_data.get("model_path")
            rvc_index_path = model_data.get("index_path")
            print(f"[Cover Generation] Using custom file model: {model_data.get('name')}")
        
    elif voice_source == 'upload_rvc':
        # Save uploaded RVC files temporarily
        rvc_model_path = str(UPLOAD_DIR / f"{job_id}_model.pth")
        async with aiofiles.open(rvc_model_path, 'wb') as f:
            content = await rvc_model_file.read()
            await f.write(content)
        
        if rvc_index_file:
            rvc_index_path = str(UPLOAD_DIR / f"{job_id}_model.index")
            async with aiofiles.open(rvc_index_path, 'wb') as f:
                content = await rvc_index_file.read()
                await f.write(content)
        print(f"[Cover Generation] Using uploaded RVC model")
    
    print(f"[Cover Generation] Job {job_id}: voice_source={voice_source}")
    
    jobs_store[job_id] = {
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "flow": PipelineFlow.FLOW_B,
        "created_at": datetime.utcnow().isoformat(),
        "progress": 0,
        "output_url": None,
        "error": None
    }
    
    background_tasks.add_task(
        process_voice_cover,
        job_id,
        str(song_path),
        None,  # No voice_sample_path - we're using models now
        voice_model_id if voice_source == 'library' else None,
        {
            "pitch_shift": pitch_shift,
            "voice_source": voice_source,
            "rvc_model_path": rvc_model_path,
            "rvc_index_path": rvc_index_path,
            "direct_model_url": direct_model_url
        }
    )
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        flow=PipelineFlow.FLOW_B,
        created_at=jobs_store[job_id]["created_at"],
        message="Voice cover generation job started"
    )


# ============================================================================
# Flow C: Combined Generation
# ============================================================================

@app.post("/api/generate/combined", response_model=JobResponse)
async def generate_combined(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    duration_ms: int = Form(60000),
    lyrics: Optional[str] = Form(None),
    conversion_strength: float = Form(0.8),
    custom_model_id: str = Form(...),
    current_user: Dict = Depends(require_auth)
):
    """
    Generate original music with custom voice (Flow C)
    
    Combines Flow A (music generation) with Flow B (voice conversion).
    Now uses saved custom models instead of voice samples.
    """
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ElevenLabs API key not configured")
    
    if not REPLICATE_API_KEY:
        raise HTTPException(status_code=500, detail="Replicate API key not configured")
    
    # Look up model - check both custom models and default models
    model_data = None
    
    # Check default models first
    for default_model in DEFAULT_VOICE_MODELS:
        if default_model["id"] == custom_model_id:
            model_data = default_model
            break
    
    # If not found in defaults, check user's custom models
    if not model_data and custom_model_id in custom_models_store:
        model_data = custom_models_store[custom_model_id]
    
    if not model_data:
        raise HTTPException(status_code=400, detail="Voice model not found")
    
    model_url = model_data.get("model_url")
    
    if not model_url:
        raise HTTPException(status_code=400, detail="Custom model has no URL")
    
    user_id = current_user["id"]
    job_id = str(uuid.uuid4())
    
    jobs_store[job_id] = {
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "flow": PipelineFlow.FLOW_C,
        "created_at": datetime.utcnow().isoformat(),
        "progress": 0,
        "output_url": None,
        "error": None
    }
    
    background_tasks.add_task(
        process_combined_generation,
        job_id,
        model_url,
        {
            "prompt": prompt,
            "duration_ms": duration_ms,
            "lyrics": lyrics,
            "conversion_strength": conversion_strength,
            "model_name": model_data.get("name", "custom")
        }
    )
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        flow=PipelineFlow.FLOW_C,
        created_at=jobs_store[job_id]["created_at"],
        message="Combined generation job started"
    )


# ============================================================================
# Job Management
# ============================================================================

@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, current_user: Dict = Depends(require_auth)):
    """Get status of a generation job"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_store[job_id]
    
    # Check ownership
    if job.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized to view this job")
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        flow=job["flow"],
        progress=job["progress"],
        output_url=job.get("output_url"),
        error=job.get("error"),
        lyrics=job.get("lyrics"),
        duration_ms=job.get("duration_ms")
    )


@app.get("/api/jobs")
async def list_jobs(current_user: Dict = Depends(require_auth)):
    """List all jobs for the current user"""
    user_id = current_user["id"]
    return {
        "jobs": [
            {
                "job_id": job_id,
                **job_data
            }
            for job_id, job_data in jobs_store.items()
            if job_data.get("user_id") == user_id
        ]
    }


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str, current_user: Dict = Depends(require_auth)):
    """Delete a job and its output"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_store[job_id]
    
    # Check ownership
    if job.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized to delete this job")
    
    # Delete output file if exists
    output_path = OUTPUT_DIR / f"{job_id}.mp3"
    if output_path.exists():
        output_path.unlink()
    
    del jobs_store[job_id]
    
    return {"message": "Job deleted successfully"}


# ============================================================================
# Custom Models Management
# ============================================================================

# Default voice models available to all users
DEFAULT_VOICE_MODELS = [
    {
        "id": "default-drake",
        "name": "Drake",
        "type": "default",
        "source": "voice-models.com",
        "model_url": "https://huggingface.co/lepifort/drakeRVC/resolve/main/drake.zip",
        "created_at": "2024-01-01T00:00:00Z",
        "is_default": True
    },
    {
        "id": "default-taylor-swift",
        "name": "Taylor Swift",
        "type": "default",
        "source": "voice-models.com",
        "model_url": "https://huggingface.co/bennetJL/TaylorSwift/resolve/main/TaylorSwift2024.zip",
        "created_at": "2024-01-01T00:00:00Z",
        "is_default": True
    },
    {
        "id": "default-kanye-west",
        "name": "Kanye West",
        "type": "default",
        "source": "voice-models.com",
        "model_url": "https://huggingface.co/TheRealheavy/KanyeCollegeDropout/resolve/main/KanyeCollegeDropout.zip",
        "created_at": "2024-01-01T00:00:00Z",
        "is_default": True
    },
    {
        "id": "default-ariana-grande",
        "name": "Ariana Grande",
        "type": "default",
        "source": "voice-models.com",
        "model_url": "https://huggingface.co/szajean/ArianaGrande/resolve/main/ARIANAGRANDE_ES_BY_SZAJEAN.zip",
        "created_at": "2024-01-01T00:00:00Z",
        "is_default": True
    },
    {
        "id": "default-the-weeknd",
        "name": "The Weeknd",
        "type": "default",
        "source": "voice-models.com",
        "model_url": "https://huggingface.co/TheRealheavy/TheWeekndStarboyEra/resolve/main/TheWeekndStarboyEra.zip",
        "created_at": "2024-01-01T00:00:00Z",
        "is_default": True
    },
]

@app.get("/api/custom-models")
async def get_custom_models(current_user: Optional[Dict] = Depends(get_current_user)):
    """Get all custom models for the current user + default models for everyone"""
    models = []
    
    # Always include default models for everyone
    for default_model in DEFAULT_VOICE_MODELS:
        models.append({
            "id": default_model["id"],
            "name": default_model["name"],
            "type": default_model["type"],
            "source": default_model["source"],
            "model_url": default_model["model_url"],
            "created_at": default_model["created_at"],
            "is_default": True,
            "has_index": False
        })
    
    # Add user's custom models if authenticated
    if current_user:
        user_id = current_user["id"]
        for model_id, model_data in custom_models_store.items():
            if model_data.get("user_id") == user_id:
                models.append({
                    "id": model_id,
                    "name": model_data.get("name", "Unknown"),
                    "type": model_data.get("type", "uploaded"),
                    "source": model_data.get("source", "uploaded"),
                    "created_at": model_data.get("created_at", ""),
                    "has_index": model_data.get("has_index", False),
                    "file_path": model_data.get("model_path", ""),
                    "model_url": model_data.get("model_url", ""),
                    "is_default": False
                })
    
    # Sort: user models first (by date), then default models
    models.sort(key=lambda x: (x.get("is_default", False), x.get("created_at", "")), reverse=False)
    return {"models": models}


@app.post("/api/custom-models/upload")
async def upload_custom_model(
    model_name: str = Form(...),
    rvc_model_file: UploadFile = File(...),
    rvc_index_file: Optional[UploadFile] = File(None),
    current_user: Dict = Depends(require_auth)
):
    """Upload an RVC model file (.pth) and optional index file (.index)"""
    user_id = current_user["id"]
    model_id = str(uuid.uuid4())
    
    # Validate file extension
    if not rvc_model_file.filename.endswith(('.pth', '.pt')):
        raise HTTPException(status_code=400, detail="Model file must be .pth or .pt format")
    
    # Create model directory
    model_dir = MODELS_DIR / model_id
    model_dir.mkdir(exist_ok=True)
    
    # Save model file
    model_path = model_dir / f"model.pth"
    async with aiofiles.open(model_path, 'wb') as f:
        content = await rvc_model_file.read()
        await f.write(content)
    
    # Save index file if provided
    index_path = None
    if rvc_index_file:
        index_path = model_dir / f"model.index"
        async with aiofiles.open(index_path, 'wb') as f:
            content = await rvc_index_file.read()
            await f.write(content)
    
    # Add to store with user_id
    custom_models_store[model_id] = {
        "user_id": user_id,
        "name": model_name,
        "type": "uploaded",
        "source": "uploaded",
        "created_at": datetime.utcnow().isoformat(),
        "model_path": str(model_path),
        "index_path": str(index_path) if index_path else None,
        "has_index": index_path is not None
    }
    
    # Save to disk
    save_custom_models()
    
    print(f"[Custom Models] User {user_id} uploaded model: {model_name} (ID: {model_id})")
    
    return {
        "model_id": model_id,
        "name": model_name,
        "message": "Model uploaded successfully"
    }


@app.delete("/api/custom-models/{model_id}")
async def delete_custom_model(model_id: str, current_user: Dict = Depends(require_auth)):
    """Delete a custom model (only if owned by current user)"""
    user_id = current_user["id"]
    
    if model_id not in custom_models_store:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check ownership
    model_data = custom_models_store[model_id]
    if model_data.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this model")
    
    # Delete files
    model_dir = MODELS_DIR / model_id
    if model_dir.exists():
        import shutil
        shutil.rmtree(model_dir)
    
    # Remove from store
    del custom_models_store[model_id]
    save_custom_models()
    
    print(f"[Custom Models] User {user_id} deleted model: {model_data.get('name')} (ID: {model_id})")
    
    return {"message": "Model deleted successfully"}


@app.get("/api/custom-models/{model_id}")
async def get_custom_model(model_id: str, current_user: Dict = Depends(require_auth)):
    """Get a specific custom model's info (only if owned by current user)"""
    user_id = current_user["id"]
    
    if model_id not in custom_models_store:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = custom_models_store[model_id]
    
    # Check ownership
    if model_data.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to view this model")
    
    return {
        "id": model_id,
        "name": model_data.get("name", "Unknown"),
        "type": model_data.get("type", "uploaded"),
        "created_at": model_data.get("created_at", ""),
        "has_index": model_data.get("has_index", False),
        "model_path": model_data.get("model_path", ""),
        "model_url": model_data.get("model_url", ""),
        "index_path": model_data.get("index_path")
    }


# Training jobs store
training_jobs_store: Dict[str, Dict[str, Any]] = {}


@app.post("/api/custom-models/save-url")
async def save_model_from_url(request: dict, current_user: Dict = Depends(require_auth)):
    """Save a model from a URL (no upload needed)"""
    user_id = current_user["id"]
    model_name = request.get("model_name")
    model_url = request.get("model_url")
    
    if not model_name or not model_url:
        raise HTTPException(status_code=400, detail="model_name and model_url are required")
    
    model_id = str(uuid.uuid4())
    
    # Store the URL with user_id
    custom_models_store[model_id] = {
        "user_id": user_id,
        "name": model_name,
        "type": "url",
        "source": "url",
        "created_at": datetime.utcnow().isoformat(),
        "model_url": model_url,
        "model_path": None,
        "index_path": None,
        "has_index": False
    }
    
    save_custom_models()
    
    print(f"[Custom Models] User {user_id} saved model from URL: {model_name} (ID: {model_id})")
    
    return {
        "model_id": model_id,
        "name": model_name,
        "message": "Model saved successfully"
    }


@app.post("/api/custom-models/train")
async def train_custom_model(
    background_tasks: BackgroundTasks,
    request: Request,
    model_name: str = Form(...),
    epochs: int = Form(50),
    sample_count: int = Form(0),
    current_user: Dict = Depends(require_auth)
):
    """
    Save voice samples for use in voice covers.
    
    NOTE: Cloud-based RVC training (Replicate train-rvc-model) is currently disabled.
    Instead, we save the user's voice samples and use them directly with the
    realistic-voice-cloning model which supports zero-shot cloning.
    
    The voice samples are:
    1. Saved locally
    2. Combined into a single reference audio
    3. Uploaded to Replicate Files API for later use
    """
    if not REPLICATE_API_KEY:
        raise HTTPException(status_code=500, detail="Replicate API key not configured")
    
    user_id = current_user["id"]
    job_id = str(uuid.uuid4())
    
    # Parse multipart form data to get all voice samples
    form = await request.form()
    voice_files = []
    
    for key, value in form.items():
        if key.startswith('voice_sample_') and hasattr(value, 'read'):
            content = await value.read()
            filename = value.filename if hasattr(value, 'filename') else f"{key}.wav"
            voice_files.append((filename, content))
            print(f"[Voice Save] Received file: {filename} ({len(content)} bytes)")
    
    if len(voice_files) == 0:
        raise HTTPException(status_code=400, detail="No voice samples uploaded")
    
    print(f"[Voice Save] User {user_id} uploaded {len(voice_files)} voice files for: {model_name}")
    
    # Save voice files
    voice_dir = MODELS_DIR / f"voice_{job_id}"
    voice_dir.mkdir(exist_ok=True)
    
    total_size = 0
    for filename, content in voice_files:
        file_path = voice_dir / filename
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        total_size += len(content)
    
    training_jobs_store[job_id] = {
        "status": "processing",
        "progress": 10,
        "message": "Processing voice samples...",
        "model_name": model_name,
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat(),
        "voice_url": None,
        "error": None,
        "voice_dir": str(voice_dir)
    }
    
    # Start processing in background
    background_tasks.add_task(
        process_voice_sample_save,
        job_id,
        model_name,
        str(voice_dir),
        voice_files,
        user_id
    )
    
    return {
        "job_id": job_id,
        "message": "Processing voice samples..."
    }


@app.get("/api/training-jobs/{job_id}")
async def get_training_job_status(job_id: str):
    """Get status of a training job"""
    if job_id not in training_jobs_store:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = training_jobs_store[job_id]
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "error": job.get("error"),
        "model_url": job.get("model_url")
    }


async def process_voice_sample_save(job_id: str, model_name: str, voice_dir: str, voice_files: list, user_id: str):
    """
    Train an RVC voice model using RunPod serverless GPU.
    
    Process:
    1. Create a ZIP file from voice samples
    2. Upload to cloud storage (Replicate Files API)
    3. Call RunPod serverless endpoint to train RVC model
    4. Poll for completion
    5. Save the trained model URL
    
    If RunPod is not configured, falls back to saving voice samples for manual training.
    """
    try:
        training_jobs_store[job_id]["status"] = "processing"
        training_jobs_store[job_id]["message"] = "Processing voice samples..."
        training_jobs_store[job_id]["progress"] = 5
        
        # Check if we have training capability
        has_runpod = bool(RUNPOD_API_KEY)
        has_replicate = bool(REPLICATE_API_KEY)
        
        if not has_replicate:
            raise Exception("Replicate API key required for file upload")
        
        # Step 1: Create a ZIP file containing all voice samples
        # Structure must be: dataset/<model_name>/<files>.wav
        import zipfile
        import io
        
        # Clean model name for folder
        clean_name = model_name.replace(" ", "_").replace("-", "_")
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
        if not clean_name:
            clean_name = "voice_model"
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, (filename, content) in enumerate(voice_files):
                # Structure: dataset/<model_name>/split_<i>.wav
                # This matches what create-rvc-dataset produces
                ext = filename.split('.')[-1] if '.' in filename else 'wav'
                new_filename = f"split_{i}.{ext}"
                zf.writestr(f"dataset/{clean_name}/{new_filename}", content)
        
        zip_buffer.seek(0)
        zip_data = zip_buffer.read()
        
        print(f"[Training] Created ZIP dataset: {len(zip_data)} bytes with {len(voice_files)} files")
        print(f"[Training] ZIP structure: dataset/{clean_name}/split_*.wav")
        
        training_jobs_store[job_id]["progress"] = 15
        training_jobs_store[job_id]["message"] = "Uploading voice samples to cloud..."
        
        # Step 2: Upload ZIP to cloud storage
        dataset_url = None
        async with aiohttp.ClientSession() as session:
            # Upload to Replicate Files API
            print(f"[Training] Uploading to Replicate Files API...")
            form_data = aiohttp.FormData()
            form_data.add_field(
                'content',
                zip_data,
                filename='voice_dataset.zip',
                content_type='application/zip'
            )
            
            try:
                async with session.post(
                    'https://api.replicate.com/v1/files',
                    headers={'Authorization': f'Bearer {REPLICATE_API_KEY}'},
                    data=form_data,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    if resp.status in [200, 201]:
                        result = await resp.json()
                        dataset_url = result.get('urls', {}).get('get')
                        print(f"[Training] Dataset uploaded: {dataset_url}")
                    else:
                        error_text = await resp.text()
                        print(f"[Training] Replicate Files API error: {resp.status} - {error_text}")
            except Exception as e:
                print(f"[Training] Upload exception: {e}")
        
        if not dataset_url:
            raise Exception("Failed to upload training data to cloud storage")
        
        training_jobs_store[job_id]["progress"] = 25
        
        # Step 3: Start training with Replicate (working version)
        async with aiohttp.ClientSession() as training_session:
            training_jobs_store[job_id]["message"] = "Starting RVC model training on cloud GPU..."
            model_url = await train_with_replicate(training_session, job_id, dataset_url, model_name)
        
        if not model_url:
            raise Exception("Training failed - no model URL returned")
        
        # Step 4: Save the trained model
        training_jobs_store[job_id]["progress"] = 95
        training_jobs_store[job_id]["message"] = "Saving trained model..."
        
        model_id = str(uuid.uuid4())
        
        custom_models_store[model_id] = {
            "user_id": user_id,
            "name": model_name,
            "type": "trained",
            "source": "trained",
            "created_at": datetime.utcnow().isoformat(),
            "model_url": model_url,
            "model_path": None,
            "index_path": None,
            "has_index": False,
            "training_method": "replicate"
        }
        
        save_custom_models()
        
        training_jobs_store[job_id]["model_url"] = model_url
        training_jobs_store[job_id]["model_id"] = model_id
        training_jobs_store[job_id]["status"] = "completed"
        training_jobs_store[job_id]["progress"] = 100
        training_jobs_store[job_id]["message"] = f"✅ Voice model '{model_name}' trained successfully! You can now use it in Voice Covers."
        
        print(f"[Training] Model trained and saved: {model_name} (ID: {model_id})")
        
        # Cleanup
        try:
            import shutil
            shutil.rmtree(voice_dir)
        except:
            pass
        
    except Exception as e:
        print(f"[Training] Error: {e}")
        import traceback
        traceback.print_exc()
        training_jobs_store[job_id]["status"] = "failed"
        training_jobs_store[job_id]["error"] = str(e)
        training_jobs_store[job_id]["message"] = f"Training failed: {str(e)}"


async def train_with_runpod(session: aiohttp.ClientSession, job_id: str, dataset_url: str, model_name: str) -> Optional[str]:
    """
    NOTE: The standard chavinlo/rvc-runpod endpoint is for INFERENCE only, not training.
    
    For actual training, you would need to:
    1. Deploy a custom Docker image with RVC training capability
    2. Or use a RunPod GPU Pod (not serverless) with RVC WebUI
    
    This function will return an error explaining the situation.
    For now, users should use pre-trained models from voice-models.com or HuggingFace.
    """
    # The depositame/rvc_runpod_serverless image is inference-only
    # It expects: model_name (HuggingFace model), audio_url, pitch, etc.
    # It does NOT support training
    
    raise Exception(
        "Cloud-based RVC training is not currently available. "
        "The RunPod endpoint supports voice conversion (inference) but not training. "
        "\n\nAlternatives:\n"
        "1. Use pre-trained models from voice-models.com (Save Model from URL)\n"
        "2. Train locally using RVC WebUI and upload the model\n"
        "3. Use a RunPod GPU Pod with RVC WebUI for training"
    )


async def train_with_replicate(session: aiohttp.ClientSession, job_id: str, dataset_url: str, model_name: str) -> Optional[str]:
    """
    Train RVC model using Replicate.
    """
    headers = {
        "Authorization": f"Bearer {REPLICATE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Use the working version (as of Feb 2025)
    version = "0397d5e28c9b54665e1e5d29d5cf4f722a7b89ec20e9dbf31487235305b1a101"
    
    # Clean model name (no spaces or special chars)
    clean_model_name = model_name.replace(" ", "_").replace("-", "_")
    clean_model_name = ''.join(c for c in clean_model_name if c.isalnum() or c == '_')
    
    train_payload = {
        "version": version,
        "input": {
            "dataset_zip": dataset_url,
            "model_name": clean_model_name,  # Required!
            "sample_rate": "48k",
            "version": "v2",
            "f0method": "rmvpe_gpu",
            "epoch": 80,
            "batch_size": "7"
        }
    }
    
    print(f"[Replicate] Starting training with version {version[:12]}...")
    print(f"[Replicate] Model name: {clean_model_name}")
    print(f"[Replicate] Dataset URL: {dataset_url}")
    
    try:
        async with session.post(
            f"{REPLICATE_BASE_URL}/predictions",
            headers=headers,
            json=train_payload
        ) as resp:
            if resp.status in [200, 201]:
                prediction = await resp.json()
                prediction_id = prediction.get("id")
                print(f"[Replicate] Training started! Prediction ID: {prediction_id}")
            else:
                error = await resp.text()
                print(f"[Replicate] Failed to start training: {error}")
                raise Exception(f"Replicate API error: {error}")
    except Exception as e:
        print(f"[Replicate] Exception: {e}")
        raise
    
    # Poll for completion
    training_jobs_store[job_id]["progress"] = 30
    training_jobs_store[job_id]["message"] = "Training model (this may take 10-20 minutes)..."
    
    for attempt in range(720):  # Up to 60 minutes
        await asyncio.sleep(5)
        
        async with session.get(
            f"{REPLICATE_BASE_URL}/predictions/{prediction_id}",
            headers=headers
        ) as resp:
            data = await resp.json()
            status = data.get("status")
            
            # Log full response for debugging on first few attempts or status changes
            if attempt < 3 or attempt % 12 == 0:
                print(f"[Replicate] Poll {attempt}: status={status}")
                if status in ["failed", "canceled"]:
                    print(f"[Replicate] Full response: {data}")
            
            if status == "succeeded":
                model_url = data.get("output")
                print(f"[Replicate] Training completed! Model: {model_url}")
                return model_url
            
            elif status in ["failed", "canceled"]:
                # Get error details - could be in different places
                error = data.get("error")
                if error is None:
                    error = data.get("logs", "")[-500:] if data.get("logs") else ""
                if not error:
                    error = f"Status: {status}, check Replicate dashboard for details"
                print(f"[Replicate] Training failed. Error: {error}")
                raise Exception(f"Training failed: {error}")
            
            # Update progress
            if attempt % 12 == 0:
                progress = min(30 + (attempt // 12) * 2, 90)
                training_jobs_store[job_id]["progress"] = progress
    
    raise Exception("Training timed out after 60 minutes")


# ============================================================================
# Voice Models
# ============================================================================

@app.get("/api/voice-models")
async def get_voice_models(my_models: bool = False):
    """Get available voice models from Kits.AI"""
    if not KITS_API_KEY:
        raise HTTPException(status_code=500, detail="Kits.AI API key not configured")
    
    models = await kits_client.get_voice_models(my_models_only=my_models)
    return {"models": models}


@app.post("/api/voice-models")
async def create_voice_model(
    name: str = Form("Custom Voice"),
    audio_file: UploadFile = File(...)
):
    """Create a new voice model from audio sample"""
    if not KITS_API_KEY:
        raise HTTPException(status_code=500, detail="Kits.AI API key not configured")
    
    audio_data = await audio_file.read()
    model_id = await kits_client.create_voice_model(audio_data, name)
    
    return {
        "model_id": model_id,
        "name": name,
        "message": "Voice model created successfully"
    }


# ============================================================================
# Prompts Library
# ============================================================================

PROMPT_LIBRARY = {
    "early_2000s_pop": {
        "name": "Early 2000s Pop Rock",
        "prompt": "Catchy early 2000s pop rock anthem, energetic female vocalist with powerful delivery, guitar-driven with jangly electric guitars and acoustic strumming, punchy drums with driving beat, bright synth accents, upbeat and radio-friendly, feel-good summer vibes, clear verse-chorus structure, 120 BPM, hooks throughout, nostalgic Y2K pop punk energy, polished production"
    },
    "indie_rock": {
        "name": "Dreamy Indie Rock",
        "prompt": "Dreamy, psychedelic, slow Indie Rock, reverb-soaked vocals, retro keys, catchy chorus, analog, phased guitars, liminal, nostalgic feeling, anthem"
    },
    "cinematic": {
        "name": "Cinematic Western",
        "prompt": "An epic track for a cowboy show, wild west, cinematic sound design, guitar twanging with awesome orchestral elements crescendoing to a powerful finale, soundtrack"
    },
    "electronic": {
        "name": "Progressive House",
        "prompt": "Progressive house track, euphoric and uplifting, featuring filtered chord stabs, rolling bassline, and crisp percussion, building energy toward a satisfying drop"
    },
    "lofi": {
        "name": "Lo-Fi Hip Hop",
        "prompt": "Chill lo-fi hip hop beat, dusty vinyl texture, mellow jazz piano chords, laid-back drums with swing, warm bass, rain sounds in background, perfect for studying and relaxation, 85 BPM"
    },
    "orchestral": {
        "name": "Epic Orchestral",
        "prompt": "Epic orchestral score, sweeping strings, powerful brass fanfares, thundering timpani, building from quiet tension to triumphant climax, cinematic and emotional"
    }
}


@app.get("/api/prompts")
async def get_prompt_library():
    """Get library of pre-built prompts"""
    return {"prompts": PROMPT_LIBRARY}


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
