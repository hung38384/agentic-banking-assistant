# Deployment Guide

This project supports three deployment levels:

1. Local Docker demo with `docker-compose.yml`
2. Production-style Docker Compose with a prebuilt image
3. GitHub Actions CD to GitHub Container Registry and optional VPS restart

## Local Docker Demo

```bash
docker compose up --build
```

The local compose file uses `LLM_PROVIDER=mock` so the demo is repeatable without external API keys.

Health check:

```bash
curl http://localhost:8000/health
```

## Production Compose

On a server, create `.env` next to `docker-compose.prod.yml`:

```env
ENVIRONMENT=production
API_KEY=replace-with-strong-api-key
LLM_PROVIDER=groq
GROQ_API_KEY=replace-with-groq-key
GROQ_MODEL=llama-3.3-70b-versatile
VECTOR_BACKEND=chroma
SEMANTIC_CACHE_ENABLED=true
HITL_ENABLED=true
MEMORY_PERSISTENCE_ENABLED=true
MEMORY_PERSIST_REQUIRES_CONSENT=true
OTEL_ENABLED=false
```

Start:

```bash
export IMAGE_NAME=ghcr.io/<owner>/<repo>:latest
docker compose -f docker-compose.prod.yml up -d
```

Check status:

```bash
docker compose -f docker-compose.prod.yml ps
curl http://localhost:8000/health
```

## GitHub Container Registry CD

The workflow `.github/workflows/cd.yml` builds and pushes images to:

```text
ghcr.io/<owner>/<repo>
```

It runs on:

- push to `main`
- version tags like `v1.0.0`
- manual `workflow_dispatch`

Required repository settings:

- GitHub Actions permissions: allow package write.
- Package visibility: private or public depending on portfolio needs.

## Optional VPS Deploy

The CD workflow has a manual `deploy_vps=true` path. Add these GitHub Secrets:

```text
VPS_HOST
VPS_USER
VPS_SSH_KEY
VPS_APP_DIR
```

The server should already have:

- Docker
- Docker Compose plugin
- a `.env` file in `VPS_APP_DIR`
- network egress to Groq/Gemini/OpenAI if using external LLMs

Manual deploy flow:

1. Open GitHub Actions.
2. Run `CD`.
3. Set `deploy_vps=true`.
4. The workflow copies `docker-compose.prod.yml`, pulls the latest GHCR image, restarts the service, and prunes old images.

## Production Hardening Checklist

- Use a strong `API_KEY`, stored only as a secret.
- Put the service behind HTTPS and a reverse proxy or API gateway.
- Restrict `/audit`, `/traces`, `/approvals`, and operational endpoints by network or stronger auth before real deployment.
- Use private LLM deployment, VPC endpoint, or bank-approved model gateway before processing real PII.
- Replace Chroma local persistence with managed Qdrant, pgvector, or a highly available vector service for production.
- Enable OpenTelemetry export with a collector endpoint.
- Configure log retention, audit retention, and PII masking according to internal policy.
- Run `scripts/end_user_smoke_test.py`, `pytest`, and evaluation suites before release.
