# TDS LLM Quiz Solver

An intelligent, automated quiz-solving application that leverages Large Language Models (LLMs) to handle complex data tasks including sourcing, preparation, analysis, and visualization.

## 📋 Overview

This project implements an API endpoint that receives quiz tasks, autonomously solves them using LLMs, and submits answers within strict time constraints. The system can handle multiple data formats, generate visualizations, and learn from incorrect attempts through an intelligent retry mechanism.

## 🎯 Key Features

- **Intelligent Question Parsing**: Uses headless browser (Playwright) to render JavaScript-heavy quiz pages
- **Multi-Format Data Processing**: Supports CSV, JSON, PDF, Excel, and audio files
- **Audio Transcription**: Converts speech to text using AI-powered transcription
- **Automated Visualization**: Generates charts (bar, line, scatter, pie, histogram) on demand
- **Adaptive Retry Logic**: Learns from incorrect attempts with feedback-driven improvements
- **Strict Time Management**: Operates within 3-minute time windows per task
- **Secure Authentication**: Validates requests using secret key verification
- **Asynchronous Architecture**: Non-blocking background processing for efficient resource usage

## 🏗️ Architecture

```
┌─────────────┐         ┌──────────────────┐         ┌─────────────┐
│   Quiz      │  POST   │   FastAPI        │  Async  │   Solver    │
│   Server    │────────>│   Endpoint       │────────>│   Engine    │
└─────────────┘         └──────────────────┘         └─────────────┘
                              │                              │
                              │                              ▼
                              │                       ┌─────────────┐
                              │                       │  Playwright │
                              │                       │  (Browser)  │
                              │                       └─────────────┘
                              │                              │
                              ▼                              ▼
                       ┌──────────────┐            ┌─────────────────┐
                       │  Response    │            │  LLM (GPT-4)    │
                       │  (200/400/   │            │  Analysis &     │
                       │   403)       │            │  Reasoning      │
                       └──────────────┘            └─────────────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │  Data Resources │
                                                  │  (CSV/PDF/JSON) │
                                                  └─────────────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │  Matplotlib     │
                                                  │  (Visualization)│
                                                  └─────────────────┘
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/tds-llm-quiz-solver.git
   cd tds-llm-quiz-solver
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   SECRET_KEY=your_secret_key_here
   AIPIPE_TOKEN=your_aipipe_token_here
   AIPIPE_URL=https://aipipe.org/openai/v1/chat/completions
   ```

### Configuration

Update the Google Form with your deployment details:
- **Email**: Your email address
- **Secret**: Must match `SECRET_KEY` in `.env`
- **API Endpoint URL**: Your deployed endpoint (e.g., `https://yourdomain.com/receive_requests`)
- **GitHub Repo URL**: Your public repository URL with MIT LICENSE

## 📦 Dependencies

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.4.0
python-dotenv>=1.0.0
httpx>=0.25.0
playwright>=1.40.0
beautifulsoup4>=4.12.0
pypdf>=3.17.0
matplotlib>=3.7.0
numpy>=1.24.0
```

## 🎮 Usage

### Running Locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### Testing Your Endpoint

Send a POST request to test the demo endpoint:

```bash
curl -X POST http://localhost:8000/receive_requests \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your_email@example.com",
    "secret": "your_secret_key",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
  }'
```

Expected response:
```json
{
  "message": "Request accepted. Solver started."
}
```

### Deployment

**Recommended platforms:**
- **Railway**: Easy deployment with automatic HTTPS
- **Render**: Free tier with persistent storage
- **Fly.io**: Global edge deployment
- **Heroku**: Simple git-based deployment
- **AWS EC2/Lambda**: Scalable cloud infrastructure

**Deployment checklist:**
- ✅ Set environment variables on platform
- ✅ Use HTTPS endpoint (required)
- ✅ Configure health check endpoint if needed
- ✅ Set appropriate timeout limits (3+ minutes)
- ✅ Enable logging for debugging

## 🔧 API Endpoints

### `POST /receive_requests`

Receives quiz tasks and initiates solving process.

**Request Body:**
```json
{
  "email": "student@example.com",
  "secret": "your_secret_key",
  "url": "https://example.com/quiz-834"
}
```

**Response Codes:**
- `200 OK`: Request accepted, solver started
- `400 Bad Request`: Invalid JSON or missing fields
- `403 Forbidden`: Invalid secret key

**Success Response:**
```json
{
  "message": "Request accepted. Solver started."
}
```

## 🧠 How It Works

### 1. **Request Reception**
   - Validates JSON payload structure
   - Verifies secret key authentication
   - Spawns asynchronous background task

### 2. **Page Analysis**
   - Launches headless Chromium browser
   - Renders JavaScript-based quiz page
   - Extracts question text and submit URL
   - Discovers downloadable resources (CSV, PDF, audio, etc.)

### 3. **Resource Processing**
   - Downloads linked files (up to 3 resources)
   - Parses CSV data (up to 300 rows)
   - Extracts text from PDFs (up to 30,000 chars)
   - Transcribes audio files using AI
   - Structures data for LLM analysis

### 4. **LLM Reasoning**
   - Constructs comprehensive prompt with:
     - Question context
     - Resource summaries
     - Previous attempt feedback (if any)
   - Requests structured JSON response
   - Validates answer format and size

### 5. **Visualization Generation** (if needed)
   - Detects visualization keywords in question
   - Extracts relevant data from resources
   - Generates chart specification via LLM
   - Creates PNG image using matplotlib
   - Encodes as base64 data URI

### 6. **Answer Submission**
   - Constructs submission payload
   - Validates payload size (< 1MB)
   - POSTs to submit endpoint
   - Handles response and feedback

### 7. **Retry Mechanism** (up to 3 attempts)
   - Analyzes failure reason from response
   - Incorporates feedback into next attempt
   - Avoids repeating incorrect approaches
   - Continues to next URL if provided

### 8. **Chain Resolution**
   - Follows next URLs until quiz completes
   - Maintains 3-minute overall timeout
   - Handles errors gracefully

## 📊 Supported Data Formats

| Format | Extensions | Processing Method |
|--------|-----------|-------------------|
| CSV | `.csv` | Pandas-style parsing, 300 rows |
| JSON | `.json` | Native JSON decoding |
| PDF | `.pdf` | Text extraction via pypdf |
| Excel | `.xlsx`, `.xls` | Planned (not yet implemented) |
| Audio | `.wav`, `.mp3`, `.m4a`, `.ogg` | AI transcription via AIPipe |
| Text | `.txt` | Direct text reading |

## 📈 Visualization Types

- **Bar Chart**: Categorical comparisons
- **Line Chart**: Trends over time
- **Scatter Plot**: Correlations and distributions
- **Pie Chart**: Proportional data
- **Histogram**: Frequency distributions

**Output format**: Base64-encoded PNG data URI
**Max size**: Under 1MB (automatic compression)

## 🔐 Security Features

- Secret key authentication for all requests
- Environment variable configuration (no hardcoded credentials)
- Input validation and sanitization
- HTTP status code enforcement (400, 403, 200)
- Payload size limits (1MB)
- Timeout constraints (3 minutes)

## 🐛 Debugging

### Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common issues:

**Issue**: `SECRET_KEY environment variable not set`
- **Solution**: Create `.env` file with `SECRET_KEY=your_secret`

**Issue**: `Playwright browser not found`
- **Solution**: Run `playwright install chromium`

**Issue**: `AIPIPE_TOKEN is not set`
- **Solution**: Add `AIPIPE_TOKEN=your_token` to `.env`

**Issue**: Timeout errors
- **Solution**: Check network connectivity, increase resource limits

**Issue**: LLM fails to return JSON
- **Solution**: Review prompt construction, check API quota

## 📝 Project Structure

```
tds-llm-quiz-solver/
├── main.py                 # FastAPI application & solver logic
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not in git)
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── LICENSE                # MIT License
└── tests/                 # Unit tests (optional)
    ├── test_solver.py
    └── test_api.py
```



### Manual Testing
1. Use the demo endpoint: `https://tds-llm-analysis.s-anand.net/demo`
2. Monitor logs for solver progress
3. Verify correct HTTP response codes
4. Test with invalid secrets/payloads

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📧 Contact

**Your Name** - 22f3000926@ds.study.iitm.ac.in

**Project Link**: [(https://github.com/22f3000926/TDS_Project2)]([https://github.com/22f3000926/TDS_Project2])

---
