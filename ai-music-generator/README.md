# 🎵 AI Music Generator

A full-stack AI music generation system that supports:
- **Flow A**: Original music generation from text prompts (ElevenLabs Eleven Music)
- **Flow B**: Voice/cover generation with custom voices (Kits.AI)
- **Flow C**: Combined - generate original music with your own voice

## 🚀 Features

- **Text-to-Music**: Generate complete songs with vocals from natural language descriptions
- **Voice Cloning**: Clone any voice with just 15-30 seconds of audio
- **Cover Generation**: Re-sing any song with a different voice
- **Composition Plans**: Fine-grained control over song structure
- **Pre-built Prompts**: Quick-start templates for various genres
- **Real-time Progress**: Track generation progress with live updates
- **Audio Player**: Preview and download generated tracks

## 📁 Project Structure

```
ai-music-generator/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # React application
│   │   └── main.jsx         # Entry point
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── Dockerfile
├── docker-compose.yml       # Docker orchestration
├── nginx.conf              # Reverse proxy config
├── .env.example            # Environment template
└── README.md
```

## 🛠️ Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose (for containerized deployment)
- ElevenLabs API key
- Kits.AI API key

### Local Development

1. **Clone and setup environment:**
```bash
cd ai-music-generator
cp .env.example .env
# Edit .env with your API keys
```

2. **Start the backend:**
```bash
cd backend
pip install -r requirements.txt
export ELEVENLABS_API_KEY=your_key_here
export KITS_API_KEY=your_key_here
python main.py
```

3. **Start the frontend:**
```bash
cd frontend
npm install
npm run dev
```

4. **Access the application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Docker Deployment

1. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

2. **Build and run:**
```bash
docker compose up --build
```

3. **Access the application:**
- Application: http://localhost
- API Docs: http://localhost/api/docs

## 📡 API Endpoints

### Music Generation (Flow A)

```bash
# Generate music from prompt
POST /api/generate/music
{
  "prompt": "Upbeat pop song with female vocals",
  "duration_ms": 60000,
  "instrumental_only": false,
  "lyrics": "optional lyrics here"
}

# Generate with composition plan
POST /api/generate/music/composition
{
  "sections": [
    {"name": "Intro", "duration_ms": 8000, "style": "..."},
    {"name": "Verse", "duration_ms": 25000, "style": "...", "lyrics": "..."}
  ],
  "global_style": "pop rock"
}
```

### Voice Model Training

Users can train custom voice models from their voice recordings:

```bash
# Train a custom voice model
POST /api/custom-models/train
Form Data:
  - model_name: string
  - voice_sample_0, voice_sample_1, ...: audio files (5-20 min total)
  - epochs: int (optional, default 50)

# Check training status
GET /api/training-jobs/{job_id}
```

**How Voice Training Works:**
1. User uploads 5-20 minutes of clean voice recordings
2. Audio is uploaded to cloud storage
3. RVC model is trained on RunPod serverless GPU
4. Trained model URL is saved for use in voice covers

### Voice Cover (Flow B)

```bash
# Generate cover with voice model
POST /api/generate/cover
Form Data:
  - voice_model_id: string
  - song_file: file
  - conversion_strength: float (0-1)
  - pitch_shift: int (-12 to 12)
```

### Combined Generation (Flow C)

```bash
# Generate original + apply custom voice
POST /api/generate/combined
Form Data:
  - prompt: string
  - duration_ms: int
  - lyrics: string (optional)
  - voice_sample: file
  - conversion_strength: float
```

### Job Management

```bash
# Get job status
GET /api/jobs/{job_id}

# List all jobs
GET /api/jobs

# Delete job
DELETE /api/jobs/{job_id}
```

### Voice Models

```bash
# List voice models
GET /api/voice-models

# Create voice model
POST /api/voice-models
Form Data:
  - name: string
  - audio_file: file
```

## 🎨 Frontend Usage

### Flow A: Original Music
1. Select or write a prompt describing your desired music
2. Adjust duration (10-300 seconds)
3. Toggle instrumental-only if needed
4. Optionally add custom lyrics
5. Click "Generate Music"

### Flow B: Voice Cover
1. Select a voice model from the library
2. Upload the song you want to cover
3. Adjust conversion strength and pitch
4. Click "Generate Cover"

### Flow C: Combined
1. Write your music prompt
2. Upload a 15-30 second voice sample
3. Adjust duration and voice strength
4. Optionally add lyrics
5. Click "Generate with Your Voice"

## 💰 Cost Estimation

| Operation | Approximate Cost |
|-----------|-----------------|
| 1-minute song (ElevenLabs) | ~$0.30-0.50 |
| Voice conversion (Kits.AI) | Included in subscription |
| Full cover generation | ~$0.50-1.00 |

## 🔧 Configuration

### RunPod Setup (for Voice Training)

To enable voice model training, you need to set up a RunPod serverless endpoint:

1. **Create a RunPod account** at [runpod.io](https://runpod.io)

2. **Deploy an RVC training endpoint:**
   - Go to Serverless > My Endpoints > New Endpoint
   - Use Docker image: `depositame/rvc_runpod_serverless:latest`
   - Or deploy from: https://github.com/chavinlo/rvc-runpod
   - Select GPU: RTX 3090 or better recommended
   - Set Max Workers: 1-3

3. **Configure environment variables:**
   ```bash
   RUNPOD_API_KEY=your_api_key_from_runpod_settings
   RUNPOD_ENDPOINT_ID=your_endpoint_id_from_serverless_dashboard
   ```

4. **Test the endpoint:**
   - Go to "Train Model" tab in the app
   - Upload voice recordings (5-20 minutes total)
   - Click "Start Training"
   - Training takes 20-60 minutes

**Estimated Costs:**
- RunPod RTX 3090: ~$0.39/hour
- 50 epochs training: ~$0.50-1.00 per model

### ElevenLabs Plans
- Free: 11 minutes/month
- Starter ($5/mo): 22 minutes
- Creator ($22/mo): 62 minutes
- Pro ($99/mo): 304 minutes

### Kits.AI Plans
- Free: 15 minutes conversions
- Converter ($11.99/mo): 15 min downloads
- Creator ($19.99/mo): 30 min downloads
- Composer ($47.99/mo): Unlimited downloads

## 🚨 Error Handling

The API handles common errors:
- **400 Bad Request**: Invalid prompt or parameters
- **401 Unauthorized**: Invalid API key
- **408 Timeout**: Job took too long
- **500 Internal Error**: API or server error

## 📝 Prompt Tips

1. **Be specific**: Include genre, mood, tempo, instruments
2. **Describe vocals**: "female vocalist", "raspy male voice"
3. **Set structure**: "verse-chorus-verse-chorus-bridge-chorus"
4. **Add context**: "radio-friendly", "cinematic", "lo-fi"
5. **Include era**: "90s grunge", "early 2000s pop rock"

## 🔒 Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive data
- Implement rate limiting in production
- Add authentication for multi-user deployments

## 📄 License

This project is created for the Scate AI Engineer case study.

## 🙏 Acknowledgments

- [ElevenLabs](https://elevenlabs.io) - Eleven Music API
- [Kits.AI](https://kits.ai) - Voice cloning and conversion API
