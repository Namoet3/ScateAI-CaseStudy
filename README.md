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
└── README.md
```

## 🛠️ Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose (for containerized deployment)
- ElevenLabs API key
- Replicate API key

### Local Development

1. **Clone and setup environment:**
```bash
cd ai-music-generator
cp .env
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
cp .env
# Edit .env with your API keys
```

2. **Build and run:**
```bash
docker compose up --build
```

3. **Access the application:**
- Application: http://localhost
- API Docs: http://localhost/api/docs



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



## 📄 License

This project is created for the Scate AI Engineer case study.

## 🙏 Acknowledgments

- [ElevenLabs](https://elevenlabs.io) - Eleven Music API
- [Replicate]
