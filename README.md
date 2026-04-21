# Final Year Project


**Privacomply-Automated compliance auditing tool to detect GDPR & PDPA violations in websites privacy policies and cookie banners**  

---

## Overview

PrivaComply is a full-stack privacy compliance tool that automatically analyzes website privacy policies against the **EU General Data Protection Regulation (GDPR)** and **Pakistan Personal Data Protection Act (PDPA) 2023**. It combines a Retrieval-Augmented Generation (RAG) pipeline powered by Anthropic Claude with a Chrome browser extension, enabling users to assess any website's privacy practices in real time.

The system evaluates privacy policies across **15 compliance categories** and **54 individual labels**, cross-referencing them against a curated legal knowledge base of 24+ regulatory documents — from primary legislation to EDPB guidelines and enforcement decisions.

---

## Features

- **Automated Policy Scraping** — Detects and extracts privacy policies from any URL, including common paths (`/privacy`, `/policy`, `/legal`, etc.) and linked PDFs
- **RAG Compliance Pipeline** — Decomposes policies by category, retrieves relevant legal precedent from a vector database, and judges compliance using Claude Haiku with prompt caching
- **Chrome Extension** — Popup and side panel UI for real-time scanning, detailed violation reports, and compliance scoring
- **Cookie Scanner** — Classifies and optionally blocks third-party tracking cookies
- **Tracker Detection** — Identifies third-party trackers and maps them to categories (analytics, advertising, etc.)
- **Dual Regulatory Coverage** — GDPR (EU) and Pakistan PDPA 2023 with cross-framework mapping
- **Multilingual UI** — English and Urdu support via i18next
- **PDF Export** — Downloadable compliance reports via jsPDF

---

## Architecture

```
Browser Extension (React + TypeScript)
        │
        │  POST /analyze  { policy_text, url, regulation }
        ▼
Backend Server (Python HTTP)
        │
        ├── Scraper ──────── Extract policy text from URL or file
        │
        ├── Decompose ─────── Split policy into 15 category excerpts
        │
        ├── Retrieve ──────── Fetch top-8 legal chunks/category from Qdrant
        │       │               (GDPR · PDPA · ePrivacy · EDPB · ICO · Enforcement)
        │
        └── Judge ────────── Claude Haiku evaluates 54 labels (3 batched calls)
                │               with prompt caching on system context
                ▼
            ComplianceReport JSON
            { score, categories[], violations[], recommendations[] }
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Browser Extension | React 19, TypeScript, Vite, TailwindCSS, Radix UI, Zustand |
| Backend API | Python 3, BaseHTTPRequestHandler |
| LLM | Anthropic Claude Haiku (prompt caching) |
| Vector Database | Qdrant (cloud-hosted, eu-central-1) |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` (768-dim) |
| Web Scraping | Playwright, httpx, BeautifulSoup4 |
| PDF Parsing | PyPDF, pdfplumber |
| Containerization | Docker Compose (Qdrant) |
| i18n | i18next (English / Urdu) |

---

## Compliance Coverage

**Regulatory Documents (4 Layers)**

| Layer | Documents |
|---|---|
| Primary Legislation | GDPR (2016/679), ePrivacy Directive, Pakistan PDPA 2023 |
| EDPB Guidelines | Consent, Transparency, Access Rights, Data Portability, Profiling, DPO, Breach Notification, and more (11 docs) |
| ICO Guidance | UK-GDPR guidance on lawful basis, special categories, children's data, and more (10 docs) |
| Enforcement Decisions | CNIL v. Google/Facebook, EDPB/Meta, DPA/WhatsApp (4 decisions) |

**Compliance Categories (15)**

Data Collection & Minimization · Lawful Basis & Consent · Purpose Limitation · Data Retention · Third-Party Sharing · User Rights (Access/Deletion/Portability) · Security Measures · Children's Data · Cookie & Tracking · Cross-Border Transfers · Data Breach Notification · Automated Decision-Making · DPO & Accountability · Marketing Communications · Special Category Data

---

## Project Structure

```
Privacomply-FYP/
├── frontend/                   # Chrome extension (React + TypeScript)
│   ├── manifest.json           # Chrome Manifest V3
│   ├── src/
│   │   ├── background/         # Service worker
│   │   ├── content/            # Content script
│   │   ├── pages/
│   │   │   ├── popup/          # Extension popup UI
│   │   │   ├── sidepanel/      # Detailed report panel
│   │   │   └── options/        # Settings page
│   │   ├── components/         # Reusable UI components
│   │   ├── store/              # Zustand state (scan, settings, trackers)
│   │   └── locales/            # i18n strings (en / ur)
│   └── package.json
│
└── backend/                    # RAG compliance pipeline (Python)
    ├── report/main.py          # CLI entry point & HTTP server
    ├── embed/pipeline.py       # Core RAG pipeline (decompose → retrieve → judge)
    ├── scraper.py              # Privacy policy web scraper
    ├── config.py               # Category/label/model configuration
    ├── ComplianceReport/       # Document ingestion pipeline
    ├── metadata/models.py      # Data models
    ├── settings/               # Legal document source registry
    ├── docker-compose.yml      # Qdrant container setup
    └── requirements.txt
```

---

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Python 3.10+
- Docker and Docker Compose
- Anthropic API key
- Qdrant Cloud instance (or local Docker)

---

### Backend Setup

**1. Start Qdrant**

```bash
cd backend
docker-compose up -d
```

**2. Install Python dependencies**

```bash
pip install -r requirements.txt
playwright install chromium
```

**3. Configure environment**

Create `backend/.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
QDRANT_URL=https://<your-instance>.cloud.qdrant.io
QDRANT_API_KEY=<your-qdrant-key>
```

**4. Ingest legal documents**

```bash
cd backend
python report/main.py ingest --regulation both
```

This downloads, parses, chunks, embeds, and uploads all regulatory documents to Qdrant. Run once — documents are cached locally in `doc_cache/`.

**5. Start the API server**

```bash
python report/main.py serve --port 8000
```

---

### Frontend Setup

**1. Install dependencies**

```bash
cd frontend
npm install
```

**2. Build the extension**

```bash
npm run build
```

**3. Load in Chrome**

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select the `frontend/dist` folder

---

## Usage

### Browser Extension

After loading the extension, navigate to any website and click the PrivaComply icon to:

- Run a compliance scan against GDPR, PDPA, or both
- View a compliance score with category breakdown
- Inspect individual violations with article references and recommendations
- Review detected cookies and trackers
- Export a full compliance report as PDF

### CLI Analysis

```bash
# Analyze a URL
python report/main.py analyze --url https://example.com --regulation gdpr

# Analyze a local policy file
python report/main.py analyze --file policy.txt --regulation both

# Output as JSON
python report/main.py analyze --url https://example.com --output-json

# Save report to file
python report/main.py analyze --url https://example.com --output-file report.json
```

### API (HTTP Server)

```
POST http://localhost:8000/analyze
Content-Type: application/json

{
  "policy_text": "...",
  "url": "https://example.com",
  "regulation": "gdpr"
}
```

```
GET http://localhost:8000/health
```

---

## Development

```bash
# Frontend dev server (extension preview)
cd frontend && npm run dev

# Lint
cd frontend && npm run lint
```

---

## License

This project was developed for academic purposes as a Final Year Project.
