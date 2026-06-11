# AMF RAG Agent

A bilingual (French/English) question-answering system over real financial filings. You ask something like *"What does LVMH disclose about TCFD?"* and it answers from the actual 2025 Universal Registration Documents that TotalEnergies, BNP Paribas, LVMH, and Airbus filed with the AMF — about 28,000 chunks of dense regulatory French and English.

![CI](https://github.com/Ryqn0/amf-rag-agent/actions/workflows/ci.yml/badge.svg)

I built this to learn, and I built it the slow way on purpose. Where most RAG tutorials hand you a framework and a `.from_documents()` call, I wanted to write the retrieval loop, the chunking, the reranking by hand first — so I'd actually understand what the framework was doing before I let it do it for me. Then, once it worked, I spent most of the effort on the part the tutorials skip: making it behave like something you could put in front of real users. Containerized, deployed to managed Kubernetes, tested in CI, and wired up with metrics so I could see what it was doing under load.

So this README is partly a tour of the system and partly a record of the decisions and dead-ends along the way.

---

## What's actually in here

| | |
|------|-------------|
| **The agent** | A LangGraph agent that decides *when* to search rather than retrieving on every turn, calls a hybrid search tool, and answers grounded in what it finds |
| **Retrieval** | Semantic search (ChromaDB) and keyword search (Elasticsearch) running together, merged, then reranked by a multilingual cross-encoder |
| **Evaluation** | A LangSmith-traced eval harness scoring faithfulness and relevance — which is how I found the agent was hallucinating, and fixed it |
| **The latency story** | The reranker was eating ~90% of each request. Moving it to GPU took a cold query from ~417s to ~21s, and warm queries to ~4s |
| **Scaling** | Kubernetes with an autoscaler that I watched climb from 2 to 6 replicas under load, then settle back down |
| **Cloud** | Ran it live on GKE in Paris behind a public IP, confirmed it served traffic, then tore it down before it could cost me anything |
| **CI/CD** | Tests run on every push; passing builds get packaged and pushed to a registry automatically |
| **Operating it** | Health and readiness probes, a Prometheus metrics endpoint, and a Grafana dashboard for latency and throughput |
| **Multi-user** | API-key auth, per-key rate limiting, and Redis-backed sessions so two people's conversations don't bleed into each other |

---

## How it fits together

```
                    ┌─────────────┐
   you ───────────► │  FastAPI    │ ◄──── API key + rate limit
                    │   /ask      │
                    └──────┬──────┘
                           │
                    ┌──────▼──────────┐
                    │  LangGraph      │  decides whether to retrieve,
                    │  agent          │  calls search as a tool, answers
                    └──────┬──────────┘
                           │ search_documents (tool)
              ┌────────────┼────────────┐
              ▼            ▼             ▼
        ┌──────────┐ ┌──────────┐ ┌───────────┐
        │ ChromaDB │ │Elastic-  │ │Cross-     │
        │(semantic)│ │search    │ │encoder    │
        │          │ │(keyword) │ │(rerank)   │
        └──────────┘ └──────────┘ └───────────┘
            hybrid retrieval → rerank → top-k → LLM

   Sessions live in Redis (TTL'd). LLM is Claude Haiku, falling back to GPT-4o-mini.
```

The retrieval is deliberately hybrid. Dense semantic search is good at meaning but misses exact terms; keyword search catches the exact terms but misses paraphrase. Running both and reranking the combined pile with a cross-encoder gets you the best of each — and the cross-encoder is what makes the top results actually good, which is also why it was the slowest part.

---

## The parts I'm proud of (with proof)

### Finding and killing the latency bottleneck

For a long time every query took 30–40 seconds and I assumed it was the LLM. It wasn't. Tracing the pipeline in LangSmith showed the cross-encoder reranker was the culprit — it was doing all its work on CPU. Moving it to GPU helped, but the first GPU build was somehow *worse* (~417s), and the trace showed why: the Docker image was pre-caching the wrong model, so every cold start re-downloaded the right one over the network. Caching the correct model dropped a cold query to ~21s and warm queries to about 4 seconds.

The lesson that stuck: measure before you optimize. I'd have spent days tuning the LLM call that wasn't the problem.

*(Screenshot: LangSmith trace, per-step latency — `docs/images/langsmith-latency.png`)*

### Watching the autoscaler actually work

I'd read about HorizontalPodAutoscalers but never seen one earn its keep. So I set one up targeting 50% CPU on a local `kind` cluster, then threw 20 parallel request loops at it and watched. CPU spiked to ~197%, replicas climbed 2 → 4 → 6, and as the pods spread the load the per-pod CPU fell back down. Then I stopped the load and watched it scale back to 2 after the cooldown. Seeing the replica count change *on its own* was the moment Kubernetes clicked for me.

*(Screenshot: `kubectl get hpa -w`, the 2→6 climb — `docs/images/hpa-autoscaling.png`)*

### Putting it on real cloud infrastructure

I deployed it to a real GKE cluster in Paris, behind a Google Cloud load balancer with a public IP, and confirmed it served the API to the open internet — then deleted the cluster so it wouldn't quietly bill me. I deliberately left the data backends (Elasticsearch, Redis) out of the cloud deployment: the point was to prove I could orchestrate and scale on managed Kubernetes, not to rebuild the whole data layer in the cloud. In production those would be managed services or StatefulSets.

*(Screenshot: the API docs served from the GKE public IP — `docs/images/gke-deployment.png`)*

### Being able to see what it's doing

You can't operate what you can't see. There's a Prometheus metrics endpoint exposing request counts (split by success/failure) and a latency histogram, Prometheus scrapes it, and Grafana turns it into p50/p95 latency and throughput. The histogram matters more than an average would — it shows the slow tail an average hides, which is exactly the kind of thing the GPU work was fixing.

*(Screenshot: Grafana latency + throughput — `docs/images/grafana-dashboard.png`)*

---

## Built with

Python 3.13, FastAPI, LangGraph and LangChain, Claude (with GPT-4o-mini as a fallback). ChromaDB for vectors, Elasticsearch for keyword search, a `sentence-transformers` cross-encoder for reranking, multilingual MiniLM for embeddings. Docker and Kubernetes (`kind` locally, GKE in the cloud), Redis for sessions, GitHub Actions for CI/CD pushing to GHCR. LangSmith, Prometheus, and Grafana for seeing inside it. `uv` for dependencies, pytest for tests.

---

## Running it yourself

You'll need Docker and an Anthropic API key (an OpenAI key is optional — it's the fallback).

```bash
# Set up your secrets
cp .env.example .env          # fill in ANTHROPIC_API_KEY, API_KEYS, etc.

# Bring the whole stack up: API, UI, Elasticsearch, Redis, Prometheus, Grafana
docker compose up -d

# Check it's ready (this actually pings Redis and Elasticsearch)
curl http://localhost:8000/ready
```

Then everything's here:

| Service | URL |
|---------|-----|
| API (Swagger) | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

To ask something:

```bash
curl -X POST http://localhost:8000/ask \
  -H "X-API-Key: <your-key>" \
  -H "Content-Type: application/json" \
  -d '{"question": "What does LVMH disclose about TCFD?"}'
```

---

## The endpoints

| Endpoint | Method | What it's for |
|----------|--------|---------|
| `/ask` | POST | Ask a question. Authenticated, rate-limited, takes an optional `session_id` to keep a conversation going |
| `/health` | GET | Liveness — is the process alive? Checks nothing external, on purpose |
| `/ready` | GET | Readiness — can it actually serve? Checks Redis and Elasticsearch |
| `/metrics` | GET | Prometheus metrics |

The split between `/health` and `/ready` is one of those things that seems pedantic until you understand it. Liveness must *not* check dependencies — if it did, and Elasticsearch went down, Kubernetes would conclude every pod was broken and kill them all, turning one outage into a crash loop. Readiness is where you check dependencies: a pod with ES down is alive (don't restart it) but not ready (don't send it traffic). I learned the difference by getting it slightly wrong first.

---

## Tests

They run on every push through GitHub Actions, on a clean Linux machine. Everything external is mocked — the search backends, the reranker model, the LLM clients — so the suite needs no live services. That clean environment is the whole point: it caught two tests that had been quietly passing locally only because a model was cached on my disk. CI ran them on a fresh machine, the cache wasn't there, and the hidden dependency fell out in the open.

```bash
uv run pytest -v
```

---

## Deploying it

There are two flavors of Kubernetes manifest. `k8s/` is the local set for `kind` — Deployment, Service, HPA, Secret — and it's what the autoscaling demo runs on. `k8s/gke/` is the cloud version: registry image paths, a `LoadBalancer` service for a public IP, real liveness/readiness probes, and a step-by-step deploy guide. I kept actual GKE deployment as a documented manual step rather than auto-deploying every commit — auto-deploy needs a cluster sitting there costing money and cloud credentials living in CI, neither of which made sense for a learning project.

There are also two Docker images: a lean CPU one (`Dockerfile.cpu`) for the Kubernetes scaling work, and a heavier GPU/CUDA one (`Dockerfile`) for the fast serving path.

---

## Where things live

```
src/amf_rag_agent/
  agent/          the LangGraph agent and its search tool
  retrieval/      embedder, ChromaDB store, Elasticsearch store, reranker
  api/            the FastAPI app — auth, rate limiting, sessions, health, metrics
  ui/             a Streamlit front-end that talks to the API like any other client
  session_store.py    Redis-backed conversation history
  config.py
tests/            the pytest suite (everything external mocked)
k8s/ , k8s/gke/   Kubernetes manifests, local and cloud
scripts/          the eval harness, data backfill, load-testing probes
.github/workflows/  CI and the build-and-push pipeline
```

---

## Honest about the edges

A few things I want to be straight about, because pretending otherwise would undercut the point of building this to learn:

The evaluation set is small and I wrote it myself, and it's scored by a sibling of the model being evaluated. A perfect score there means the system is internally consistent, not that it's correct — a real adversarial eval built by a domain expert would score lower, and that's fine. It's a starting point, not a verdict.

The cloud deployment was API-only. Scaling, load balancing, the public endpoint — all real and all demonstrated. But end-to-end answering in the cloud would need the data backends running there too, which I left for a hypothetical production version.

And the honest framing overall: this is a project built to production *practices*, not a production *service*. I wasn't trying to launch a product. I was trying to understand every layer well enough to build it deliberately — and to leave behind something that shows that understanding.