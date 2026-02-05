# Supabase RAG MCP Server

A Model Context Protocol (MCP) server written in Go that provides RAG (Retrieval-Augmented Generation) capabilities backed by Supabase (PostgreSQL + pgvector) and OpenAI embeddings.

This server is designed to be multi-tenant, requiring a tenant ID for all requests to ensure data isolation.

## Prerequisites

- **Go**: Version 1.25 or higher.
- **Supabase Project**: A configured Supabase project with:
  - The `pgvector` extension enabled.
  - A `documents` table with `content` (text), `embedding` (vector), and `tenant_id` columns.
- **OpenAI API Key**: For generating embeddings.

## Configuration

The application is configured via environment variables. You can set these in your shell or use a `.env` file.

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SUPABASE_DB_URL` | The PostgreSQL connection string for your Supabase database. **Important:** Use the **Session Mode** connection string (port 5432) for this persistent server, not the Transaction Mode (port 6543). | `postgres://user:password@aws-0-us-east-1.pooler.supabase.com:5432/postgres` |
| `OPENAI_API_KEY` | Your OpenAI API key used to generate embeddings for search queries. | `sk-...` |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | The HTTP port the server will listen on. | `8080` |

## Quick Start (Docker Compose) [Recommended]

The easiest way to run the server is using Docker Compose.

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd mcp-server
    ```

2.  **Configure Environment**:
    Copy the example environment file and fill in your values.
    ```bash
    cp .env.example .env
    ```
    Edit `.env` and add your `SUPABASE_DB_URL` and `OPENAI_API_KEY`.

3.  **Run the Server**:
    ```bash
    docker compose up --build
    ```
    The server will start on port `8080` (or the port defined in your `.env`).

## Manual Installation & Run

If you prefer to run Go directly on your machine:

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd mcp-server
    ```

2.  **Install dependencies**:
    ```bash
    go mod tidy
    ```

3.  **Set Environment Variables**:
    You can export them directly or use a tool to load them.
    ```bash
    export SUPABASE_DB_URL="postgres://user:pass@host:5432/db"
    export OPENAI_API_KEY="sk-..."
    ```

4.  **Run the Server**:
    ```bash
    go run main.go
    ```
    Or build the binary:
    ```bash
    go build -o mcp-server
    ./mcp-server
    ```

## Usage

This server implements the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). It exposes a `rag-search` tool.

### Endpoints

-   SSE Endpoint: `/sse`
-   Messages Endpoint: `/messages`

### Content Isolation (Multi-Tenancy)

**Crucial**: All requests to the MCP server **MUST** include the `X-Tenant-ID` header. This header is used to scope all database queries to a specific tenant, ensuring that users can only retrieve documents belonging to their organization.

**Header:**
```
X-Tenant-ID: <uuid-of-tenant>
```

### Available Tools

#### `rag-search`
Searches the knowledge base for relevant documents based on a semantic query.

-   **Input**:
    -   `query` (string) - The search term or question.
    -   `limit` (number, default: 5) - Maximum number of results.
    -   `neighbor_count` (number, default: 2) - Number of neighboring chunks to fetch around each match (Context Expansion).
-   **Output**: A formatted string containing relevant document segments, each followed by a **Source Deep Link** (`https://notion.so/...`) to the specific block in Notion.

## Development

-   **Database Pooling**: The server is configured to use `pgxpool` with a conservative max connection limit (20) to play nicely with Supabase's limits.
-   **Embeddings**: Defaults to `text-embedding-3-small`. Ensure your database vectors are compatible (1536 dimensions).


# TODO:

- Add tunable hyperparameters for the RAG search (e.g., embedding model configuration).
- Add authentication (either an API key or OAuth), Streamable HTTPS too. 
