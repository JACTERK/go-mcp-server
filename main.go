package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"github.com/pgvector/pgvector-go"
	pgxvector "github.com/pgvector/pgvector-go/pgx"
	"github.com/sashabaranov/go-openai"
)

// ContextKey is a custom type for context keys to avoid collisions
type ContextKey string

const (
	TenantIDKey ContextKey = "tenantID"
)

func main() {
	// Initialize Database Connection
	dbURL := os.Getenv("SUPABASE_DB_URL")
	if dbURL == "" {
		log.Fatal("SUPABASE_DB_URL is required")
	}

	// Use pgxpool for best performance with Supabase
	// For persistent servers (like this one), use Supabase Session Mode (Port 5432).
	// Transaction Mode (Port 6543) is preferred for serverless functions (AWS Lambda, Vercel).
	dbConfig, err := pgxpool.ParseConfig(dbURL)
	if err != nil {
		log.Fatalf("Unable to parse database config: %v", err)
	}

	// Connection Strategy:
	// Limit the pool size to prevent exhausting Supabase connection limits.
	// A typical Supabase free tier limit is 60 connections; standard is ~200-500.
	// We conservatively set this to 20 to allow for other clients.
	dbConfig.MaxConns = 20
	dbConfig.MinConns = 2
	dbConfig.MaxConnLifetime = 1 * time.Hour
	dbConfig.MaxConnIdleTime = 30 * time.Minute

	// Register pgvector types
	dbConfig.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		return pgxvector.RegisterTypes(ctx, conn)
	}

	pool, err := pgxpool.NewWithConfig(context.Background(), dbConfig)
	if err != nil {
		log.Fatalf("Unable to create connection pool: %v", err)
	}
	defer pool.Close()

	// Initialize OpenAI Client (for Embedding Generation)
	openAIToken := os.Getenv("OPENAI_API_KEY")
	if openAIToken == "" {
		log.Println("Warning: OPENAI_API_KEY is missing. RAG search will fail.")
	}
	aiClient := openai.NewClient(openAIToken)

	// Create MCP Server
	s := server.NewMCPServer(
		"Supabase RAG Server",
		"1.0.0",
		server.WithToolCapabilities(true),
	)

	// Define Tools

	// Define the 'rag-search' Tool
	ragTool := mcp.NewTool("rag-search",
		mcp.WithDescription("Search the knowledge base for relevant documents."),
		mcp.WithString("query",
			mcp.Required(),
			mcp.Description("The search query to find relevant information."),
		),
		mcp.WithNumber("limit",
			mcp.Description("The maximum number of documents to retrieve (default: 5)."),
		),
	)

	// Add Tool Handler
	s.AddTool(ragTool, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		// Extract arguments
		query, err := request.RequireString("query")
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}

		limit := request.GetInt("limit", 5)
		if limit <= 0 {
			limit = 5
		}

		// Security: Retrieve Tenant ID from Context
		tenantID, ok := ctx.Value(TenantIDKey).(string)
		if !ok || tenantID == "" {
			return mcp.NewToolResultError("Unauthorized: Missing X-Tenant-ID header"), nil
		}

		// A. Generate Embedding for the query
		// We use text-embedding-3-small as a standard default. Ensure your DB vectors match this dimension (1536).
		embReq := openai.EmbeddingRequest{
			Input: []string{query},
			Model: openai.SmallEmbedding3,
		}
		embResp, err := aiClient.CreateEmbeddings(ctx, embReq)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to generate embedding: %v", err)), nil
		}
		vector := embResp.Data[0].Embedding

		// B. Query Supabase (pgvector)
		// We use the <=> operator for Cosine Distance (standard for text similarity).
		// We filter strictly by tenant_id to ensure isolation.
		sql := `
			SELECT content 
			FROM documents 
			WHERE tenant_id = $1 
			ORDER BY embedding <=> $2 
			LIMIT $3`

		// pgx can automatically encode the float32 slice into the vector type
		rows, err := pool.Query(ctx, sql, tenantID, pgvector.NewVector(vector), limit)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Database query failed: %v", err)), nil
		}
		defer rows.Close()

		var results string
		for rows.Next() {
			var content string
			if err := rows.Scan(&content); err != nil {
				continue
			}
			results += fmt.Sprintf("---\n%s\n", content)
		}

		if results == "" {
			return mcp.NewToolResultText("No relevant documents found."), nil
		}

		return mcp.NewToolResultText(results), nil
	})

	// Middleware to extract X-Tenant-ID and inject into Context
	authMiddleware := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			tenantID := r.Header.Get("X-Tenant-ID")

			// Optional: Fail early if header is missing, or let the tool handle it.
			// Here we pass it through.
			ctx := context.WithValue(r.Context(), TenantIDKey, tenantID)

			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}

	// 6. Set up Streamable HTTP Server (SSE) with Middleware
	// We need a public URL for the client (LibreChat) to reach the messages endpoint.
	// In Docker, this MUST be reachable from the client container.
	publicURL := os.Getenv("PUBLIC_URL")
	if publicURL == "" {
		publicURL = "http://localhost:8080"
		log.Println("Warning: PUBLIC_URL not set, defaulting to http://localhost:8080")
	}

	// Configure SSEServer
	sseServer := server.NewSSEServer(s,
		server.WithBaseURL(publicURL),
		server.WithMessageEndpoint("/messages"), // Match the mux path below
	)

	// Middleware for Logging
	loggingMiddleware := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			log.Printf("Received %s request for %s from %s", r.Method, r.URL.Path, r.RemoteAddr)

			// Log headers
			log.Printf("Headers: %v", r.Header)

			next.ServeHTTP(w, r)

			log.Printf("Completed %s request for %s in %v", r.Method, r.URL.Path, time.Since(start))
		})
	}

	// 7. Start HTTP Server
	mux := http.NewServeMux()
	mux.Handle("/sse", loggingMiddleware(authMiddleware(sseServer.SSEHandler())))
	mux.Handle("/messages", loggingMiddleware(authMiddleware(sseServer.MessageHandler()))) // Streamable HTTP requires a messages endpoint

	// Health check endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	fmt.Printf("MCP Server listening on port %s...\n", port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		fmt.Printf("Server error: %v\n", err)
	}
}
