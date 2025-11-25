# Story Architect

**AI-Powered Multi-Agent Pipeline for Script-to-Shot-to-Image Conversion**

Story Architect is a production-ready system that transforms story concepts into fully generated cinematic videos. Powered by Google Gemini models, it uses a specialized 11-agent pipeline to generate screenplays, break them into shots, produce consistent character-driven images, generate videos, and intelligently edit them into a final cinematic product.

## Features

### Phase 1: Script-to-Shot Conversion
- **Screenplay Generation**: Transform loglines into dialogue-driven screenplays
- **Scene Breakdown**: Verbose location and character descriptions
- **Shot Planning**: Detailed shot descriptions with first-frame and animation prompts
- **Shot Grouping**: Parent/child relationships for optimized generation

### Phase 2: Image Generation & Verification
- **Character Creation**: Generate consistent 1024x1024 character portraits with grid layouts
- **Parent Shot Images**: Transform character grids into cinematic parent shots
- **Child Shot Images**: Generate consistent child shots using parent references
- **AI Verification**: Automated quality assessment with retry logic
- **Multi-Format Export**: HTML (single/multi-part), Notion ZIP, complete archives

### Phase 3: Video Generation & Intelligent Editing (NEW)
- **Video Generation**: Transform static images into 4-8 second video clips with dialogue
- **Audio Analysis**: WhisperX-powered word-level timestamp detection
- **Intelligent Editing**: AI-driven Edit Decision Lists (EDL) using Gemini 2.5 Flash
- **Cinematic Transitions**: J-cuts and L-cuts for natural conversation flow
- **Scene Assembly**: Automated concatenation with crossfade transitions
- **Master Timeline**: Final edited video ready for distribution

### Core Features
- **Flexible Entry Points**: Start from a logline or bring your own screenplay
- **Session Management**: Save progress and resume from any agent in the pipeline
- **Smart Error Handling**: Automatic retry with feedback loop (up to 3 attempts) plus manual intervention
- **Modular Design**: Each agent can be customized independently via prompt templates
- **Interactive GUI**: Modern Next.js 14 interface with real-time updates via WebSocket
- **CLI Support**: Full command-line interface for automation

## Architecture

### 11-Agent Pipeline

**Phase 1: Script-to-Shot (Agents 1-4)**

**Agent 1: Screenplay Generator**
- Input: Logline, story concept, or rough script
- Output: Dialogue-driven screenplay with proper formatting
- Format: Plain text

**Agent 2: Scene Breakdown**
- Input: Screenplay
- Output: Scenes with verbose location and character descriptions
- Features: Hybrid subscene structure with character entrance tracking
- Format: JSON

**Agent 3: Shot Breakdown**
- Input: Scene breakdown
- Output: Individual shots with three components:
  - Shot Description: What happens in the shot
  - First Frame: Verbose visual description for image generation
  - Animation: Instructions for animating the first frame
- Format: Strict JSON schema with validation

**Agent 4: Shot Grouping**
- Input: Shot breakdown
- Output: Parent/child shot relationships in nested hierarchy
- Purpose: Optimize generation by grouping similar shots
- Features: Cross-scene grouping, multi-level nesting
- Format: Nested JSON

**Phase 2: Image Generation & Verification (Agents 5-9)**

**Agent 5: Character Creation**
- Input: Scene breakdown (Agent 2 output)
- Output: 1024x1024 character portraits + combination grids
- Model: Gemini 2.5 Flash Image generation
- Features: Consistent character design, grid layouts for references
- Storage: Images saved to session directory

**Agent 6: Parent Shot Image Generation**
- Input: Agent 4 output (parent shots) + character grids
- Output: Generated parent shot images (1024x1024 or 1280x768)
- Features: Uses character grids as visual references
- Model: Gemini 2.5 Flash Image generation

**Agent 7: Parent Shot Verification**
- Input: Generated parent shot images + shot metadata
- Output: Verification results with quality scores
- Features: Multimodal AI review, retry logic, soft failure mode
- Model: Gemini 2.5 Pro vision analysis

**Agent 8: Child Shot Image Generation**
- Input: Agent 4 output (child shots) + parent shot images + character grids
- Output: Generated child shot images
- Features: Uses parent shots and character grids for consistency
- Model: Gemini 2.5 Flash Image generation

**Agent 9: Child Shot Verification**
- Input: Generated child shot images + parent references
- Output: Verification results with consistency assessment
- Features: Cross-references parent shots for continuity
- Model: Gemini 2.5 Pro vision analysis

**Phase 3: Video Generation & Editing (Agents 10-11)**

**Agent 10: Video Dialogue Generator**
- Input: Parent and child shot images + scene/shot metadata
- Output: 4-8 second video clips with dialogue and animation
- Features: Converts static images to video using Veo 3.1 or FAL AI
- Model: Google Veo 3.1 Fast Generate (Vertex AI) or FAL AI video models
- Storage: Videos saved to session directory with metadata

**Agent 11: Intelligent Video Editor**
- Input: Generated videos + scene/shot structure
- Output: Master video timeline + per-scene videos
- Features:
  - WhisperX audio analysis for word-level timestamps
  - Gemini 2.5 Flash generates Edit Decision Lists (EDL)
  - FFmpeg-powered editing with J/L cuts
  - Automated silence trimming and pacing optimization
  - Scene-to-scene crossfade transitions
- Model: Gemini 2.5 Flash (EDL generation) + WhisperX (audio analysis)
- Tools: FFmpeg, WhisperX

### Tech Stack

**Backend:**
- Python 3.10+ with FastAPI
- SQLAlchemy (async) with SQLite database
- Google Gemini 2.5 Pro (text generation, vision analysis)
- Google Gemini 2.5 Flash (EDL generation, image generation)
- Google Veo 3.1 (video generation via Vertex AI)
- WhisperX (audio analysis and speech recognition)
- FFmpeg (video editing and processing)
- Pydantic v2 (validation and schemas)
- WebSocket support for real-time progress updates

**Frontend:**
- Next.js 14 (App Router)
- React 18 with TypeScript
- TanStack Query (React Query) for data fetching
- Radix UI + shadcn/ui components
- Tailwind CSS styling
- Zustand for state management

## Installation

### Prerequisites

- Python 3.10 or higher
- **FFmpeg**: Required for video editing (Agent 11)
  - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or install via [Chocolatey](https://chocolatey.org/): `choco install ffmpeg`
  - **macOS**: `brew install ffmpeg`
  - **Linux (Ubuntu/Debian)**: `sudo apt-get install ffmpeg`
  - Verify installation: `ffmpeg -version`
- Google Gemini API key (get one at https://aistudio.google.com/apikey)
- Optional: Google Cloud project with Vertex AI enabled (for enterprise deployments, Agent 10 video generation)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ShadoW-Shinigami/story-architect.git
cd story-architect
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API key
# GEMINI_API_KEY=your_api_key_here
```

5. (Optional) Customize `config.yaml` for agent behavior, output paths, and model settings

## Usage

### Quick Start

**Terminal 1 - Backend (FastAPI):**
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Backend API: `http://localhost:8000`
API Docs (Swagger): `http://localhost:8000/docs`

**Terminal 2 - Frontend (Next.js):**
```bash
cd frontend
npm install  # First time only
npm run dev
```
Frontend: `http://localhost:3000`

### API Documentation
Interactive API documentation is available at `http://localhost:8000/docs` where you can:
- View all available endpoints
- Test API calls directly from the browser
- See request/response schemas
- Understand WebSocket event structure

### WebSocket Connection
The frontend automatically connects to `ws://localhost:8000/ws` for real-time progress updates during pipeline execution

### Output Structure

All outputs saved to `outputs/projects/<session_id>/`:

**Phase 1 Outputs:**
- `session_state.json` - Complete session state
- `agent_1_output.txt` - Generated screenplay
- `agent_2_output.json` - Scene breakdown
- `agent_3_output.json` - Shot breakdown with visual prompts
- `agent_4_output.json` - Grouped shot hierarchy

**Phase 2 Outputs:**
- `agent_5_output.json` - Character data with image paths
- `agent_6_output.json` - Parent shot image paths and metadata
- `agent_7_output.json` - Parent verification results
- `agent_8_output.json` - Child shot image paths and metadata
- `agent_9_output.json` - Child verification results
- `characters/` - Generated character portraits and grids
- `parent_shots/` - Generated parent shot images
- `child_shots/` - Generated child shot images

## Configuration

### API Authentication

**Option 1: Direct API (Recommended for development)**
```bash
# .env
GEMINI_API_KEY=your_api_key_here
```

**Option 2: Vertex AI (Recommended for production)**
```bash
# .env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=google-credentials.json
```

Set `use_vertex_ai: true` in `config.yaml`

### Agent Customization

Edit `config.yaml` to adjust agent behavior:
```yaml
agents:
  agent_1:
    temperature: 0.8          # Creativity (0.0-1.0)
    max_output_tokens: 32000  # Max output length
    enabled: true             # Enable/disable

  agent_5:
    temperature: 0.7
    max_output_tokens: 32000
    enabled: true
```

### Prompt Templates

All prompts are in `prompts/` as `.txt` files:
- Use `{input}` placeholder for input data
- Changes take effect immediately
- No code modification needed

## Project Structure

```
story-architect/
├── backend/                          # FastAPI Backend
│   ├── app/
│   │   ├── agents/                   # 11 Agent implementations
│   │   │   ├── agent_1_screenplay.py to agent_11_video_edit.py
│   │   │   ├── base_agent.py
│   │   │   └── factory.py
│   │   ├── api/routes/               # REST API endpoints
│   │   │   ├── sessions.py, pipeline.py, agents.py
│   │   │   ├── queue.py, files.py, websocket.py
│   │   ├── core/                     # Core components
│   │   │   ├── gemini_client.py, pipeline.py, config.py
│   │   │   ├── queue_manager.py, progress_tracker.py
│   │   ├── models/                   # SQLAlchemy models
│   │   ├── schemas/                  # Pydantic schemas
│   │   ├── utils/                    # Utilities
│   │   │   ├── image_utils.py, audio_analyzer.py
│   │   │   ├── ffmpeg_builder.py, vertex_veo_helper.py
│   │   ├── database.py, main.py
│   ├── config.yaml                   # Agent configuration
│   ├── requirements.txt, pyproject.toml
│   ├── prompts/                      # Agent prompt templates
│   └── .env.example                  # Backend env template
│
├── frontend/                         # Next.js 14 Frontend
│   ├── src/
│   │   ├── app/                      # Next.js App Router
│   │   ├── components/               # React components
│   │   │   ├── ui/                   # shadcn/ui
│   │   ├── hooks/, stores/, types/
│   ├── package.json, next.config.js
│   ├── tailwind.config.ts
│   └── .env.local.example            # Frontend env template
│
├── .gitignore
├── .env.example                      # Root env template
└── README.md
```

## Example Workflow

```
1. Input: "A detective discovers a conspiracy in a futuristic city"
   ↓ Agent 1
2. Output: Full dialogue-driven screenplay (15 scenes)
   ↓ Agent 2
3. Output: 15 scenes with verbose character/location descriptions
   ↓ Agent 3
4. Output: 45 shots with first-frame prompts and animations
   ↓ Agent 4
5. Output: 12 parent shots, 33 child shots (hierarchical grouping)
   ↓ Agent 5
6. Output: 8 character portraits + 4 combination grids (1024x1024)
   ↓ Agent 6
7. Output: 12 parent shot images generated (1280x768)
   ↓ Agent 7
8. Output: Verification results (11/12 passed, 1 retry succeeded)
   ↓ Agent 8
9. Output: 33 child shot images generated (consistent with parents)
   ↓ Agent 9
10. Output: Final verification (32/33 passed, 1 soft fail documented)
```

Result: Complete cinematic sequence with 45 generated images ready for video generation.

## Troubleshooting

**API Key Issues**
```
Error: Google API key not found
```
Ensure `.env` file exists with valid `GEMINI_API_KEY`

**Rate Limits**
```
Error: Quota exceeded / Rate limit reached
```
Enable billing at https://aistudio.google.com/api-keys for higher limits (free tier: 15 RPM, paid: 1000+ RPM)

**Import Errors**
```
ModuleNotFoundError: No module named 'google.genai'
```
Activate virtual environment and run `pip install -r requirements.txt`

**Image Generation Errors**
```
Error: Image generation failed
```
- Check API key has image generation permissions
- Verify billing is enabled (image generation requires paid tier)
- Review Agent 5-9 logs for specific error messages

**Agent Validation Failures**
- System auto-retries up to 3 times with feedback
- Check error messages in GUI
- Review intermediate outputs in session history
- Manually edit outputs and resume if needed

**Export File Size Issues**
- Large HTML exports auto-split into multiple parts (>50MB)
- Use Notion ZIP export for better portability
- Use Complete Archive for full backup

## Billing & Rate Limits

The Gemini API tier (free vs paid) is determined by your Google AI Studio billing setup, not by code configuration.

**Free Tier:**
- 15 requests per minute
- Limited features
- **No image generation support**
- Data may be used to improve Google products

**Paid Tier (Required for Phase 2):**
- 1000+ requests per minute (model-dependent)
- **Full image generation access** (Gemini 2.5 Flash Image)
- Context caching, batch API
- Data NOT used for model improvement
- Enable at: https://aistudio.google.com/api-keys

**Note:** Phase 2 (Agents 5-9) requires paid tier for image generation.

See pricing: https://ai.google.dev/gemini-api/docs/pricing

## Known Limitations

- Video generation and intelligent editing now available (Agents 10-11 with Veo 3.1 + WhisperX)
- Single model support (Gemini 2.5 Pro/Flash Image)
- Manual JSON editing required for session modifications
- No batch processing
- Image generation requires paid API tier

## Future Roadmap

- **Video generation integration** (animate generated images)
- **Interactive image editing** (regenerate specific shots)
- Character/location database with consistency tracking
- Batch processing for multiple projects
- Multi-model support (Stable Diffusion, Midjourney, etc.)
- Advanced editing UI with visual timeline
- Shot-to-video animation

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

Built with Google Gemini 2.5 Pro, Gemini 2.5 Flash, Google Veo 3.1, FastAPI, and Next.js 14

Developed with assistance from Claude Code
