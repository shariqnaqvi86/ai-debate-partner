# Debate Coach (Docker-First)

AI-powered policy debate simulator and coaching dashboard (Streamlit) with persona-based debate responses, evidence-aware routing, and Ethos/Logos/Pathos scoring.

## Quick Start (Beginner-Friendly)
This project is intended to be run with Docker for reproducibility.

1. Install Docker Desktop.
2. Clone this repository.
3. Open Docker Desktop and wait until it shows as running.
4. In the project folder, run:

```bash
docker compose up --build
```

5. Open `http://localhost:8501` in your browser.

To stop:

```bash
docker compose down
```

## If Docker Fails to Start
If you see:
`Cannot connect to the Docker daemon ... Is the docker daemon running?`

Run:

```bash
open -a Docker
docker info
```

Then retry:

```bash
docker compose up --build
```

## What This App Does
- Runs a live chat debate UI in Streamlit.
- Supports multiple personas (public health officials and state legislators).
- Detects/maintains debate topic from conversation.
- Uses approved-source-aware evidence routing.
- Scores substantive turns on Ethos, Logos, and Pathos.
- Exports logs and session transcripts.

## Docker Files in This Repo
- [`Dockerfile`](Dockerfile): container image definition.
- [`docker-compose.yml`](docker-compose.yml): local multi-container/runtime config.
- [`.dockerignore`](.dockerignore): excludes non-runtime files from image build context.

## Runtime Data
The app writes runtime artifacts to `data/`:
- `data/logs.jsonl`
- `data/llm_calls.jsonl`
- `data/sessions/*.json`

`docker-compose.yml` mounts `./data` into the container, so sessions persist across restarts.

## Reproduce on Any New Machine
1. Install Docker Desktop.
2. Clone repo:

```bash
git clone <your-repo-url>
cd debate_coach
```

3. Start Docker Desktop.
4. Build and run:

```bash
docker compose up --build
```

5. Open `http://localhost:8501`.
6. Stop with `docker compose down` when done.

## Troubleshooting
- Port `8501` already in use:
  - Stop the other process or change port mapping in `docker-compose.yml`.
- Build fails during dependency install:
  - Retry with a clean build:
  ```bash
  docker compose build --no-cache
  docker compose up
  ```
- App container starts but exits quickly:
  - Check logs:
  ```bash
  docker compose logs -f
  ```
