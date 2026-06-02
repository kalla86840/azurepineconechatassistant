# Azure Pinecone Chat

This repository deploys a browser-based retrieval-augmented generation chat app with Azure DevOps, Azure App Service, OpenAI, and Pinecone.

The deployed homepage is a public chat URL:

```text
https://azure-pinecone-chat-86840.azurewebsites.net
```

The Azure DevOps pipeline creates or validates a Pinecone serverless index with exactly `1024` dimensions, ingests the bundled `.txt` files in `docs/`, builds the Docker image in Azure Container Registry, deploys the app to Azure App Service, and waits for `/health`.

## Required Setup

Create an Azure DevOps service connection named `AIOPs`, or change `azureServiceConnection` in `azure-pipelines.yml`.

Add these secret pipeline variables:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI API key for embeddings and chat answers |
| `PINECONE_API_KEY` | Pinecone API key for index creation, ingestion, and retrieval |

The default App Service name is `azure-pinecone-chat-86840`. App Service names are globally unique, so change `webAppName` if that hostname is already taken.

## Pipeline

The pipeline in `azure-pipelines.yml` runs on `main`.

1. Install Python dependencies and run tests.
2. Create the `azure-pinecone-chat-1024` Pinecone serverless index if it does not exist.
3. Reject an existing index if its dimension is not `1024`.
4. Embed and ingest sample data from `docs/`.
5. Build the container in Azure Container Registry.
6. Deploy the container to a Linux Azure App Service plan.
7. Print the public chat webpage URL after `/health` responds.

Pinecone defaults:

```text
PINECONE_INDEX=azure-pinecone-chat-1024
PINECONE_NAMESPACE=knowledge
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_EMBEDDING_DIMENSIONS=1024
```

`PINECONE_HOST` is discovered after index creation and passed to App Service automatically.

## Local Run

Copy `.env.example` to `.env`, set the API keys, bootstrap the index, ingest the data, and start the app:

```powershell
python -m app.pinecone_client --cloud aws --region us-east-1
$env:PINECONE_HOST = "<host printed by the previous command>"
python -m app.embeddings --docs-dir docs
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

## Routes

| Route | Purpose |
| --- | --- |
| `GET /` | Browser chat webpage |
| `GET /health` | Deployment health check |
| `POST /rag` | Retrieve Pinecone context and answer with OpenAI |
| `POST /query` | Semantic Pinecone search |
| `POST /memory` | Store a memory record |
| `POST /memory/search` | Search memory records |
| `GET /docs` | FastAPI interactive API documentation |

Example RAG request:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/rag" `
  -ContentType "application/json" `
  -Body '{"question":"What can this assistant answer?","top_k":5}'
```

## Azure OpenAI Option

The default deployment uses the OpenAI API directly. To use an Azure OpenAI resource instead, also configure `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_VERSION`. Azure deployment names can be supplied through the existing `OPENAI_MODEL` and `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` settings.
