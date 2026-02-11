package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
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
	transport := flag.String("transport", "http", "Transport mode: 'http' or 'stdio'")
	flag.Parse()

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
		mcp.WithString("tenant_id",
			mcp.Description("The Tenant ID (UUID). Required if not provided via X-Tenant-ID header (e.g. in stdio mode)."),
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

		// Security: Retrieve Tenant ID from Context or Argument
		tenantID, ok := ctx.Value(TenantIDKey).(string)
		if !ok || tenantID == "" {
			// Try getting it from arguments (for stdio)
			tenantID = request.GetString("tenant_id", "")
		}

		if tenantID == "" {
			return mcp.NewToolResultError("Unauthorized: Missing tenant_id. Must be provided via X-Tenant-ID header or 'tenant_id' argument."), nil
		}

		// A. Generate Embedding for the query
		embReq := openai.EmbeddingRequest{
			Input: []string{query},
			Model: openai.SmallEmbedding3,
		}
		embResp, err := aiClient.CreateEmbeddings(ctx, embReq)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to generate embedding: %v", err)), nil
		}
		vector := embResp.Data[0].Embedding

		// B. Vector Search — query child documents only
		sqlSearch := `
			SELECT id, content, metadata, parent_id,
			       embedding <=> $2 AS distance
			FROM documents
			WHERE tenant_id = $1
			  AND doc_type = 'child'
			ORDER BY embedding <=> $2
			LIMIT $3`

		type ChildHit struct {
			ID       string
			Content  string
			Metadata map[string]interface{}
			ParentID string
			Distance float64
		}

		rows, err := pool.Query(ctx, sqlSearch, tenantID, pgvector.NewVector(vector), limit)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Database query failed: %v", err)), nil
		}
		defer rows.Close()

		var hits []ChildHit
		for rows.Next() {
			var h ChildHit
			var metadataJSON []byte
			if err := rows.Scan(&h.ID, &h.Content, &metadataJSON, &h.ParentID, &h.Distance); err != nil {
				log.Printf("Error scanning child row: %v", err)
				continue
			}
			// Parse JSONB metadata
			if err := json.Unmarshal(metadataJSON, &h.Metadata); err != nil {
				log.Printf("Error parsing metadata for child %s: %v", h.ID, err)
				continue
			}
			hits = append(hits, h)
		}
		rows.Close()

		if len(hits) == 0 {
			return mcp.NewToolResultText("No relevant documents found."), nil
		}

		// C. Fetch parent documents (deduplicated)
		type ParentDoc struct {
			ID       string
			Content  string
			Metadata map[string]interface{}
		}
		parentCache := make(map[string]*ParentDoc)

		sqlParent := `
			SELECT id, content, metadata
			FROM documents
			WHERE id = $1`

		for _, hit := range hits {
			if _, exists := parentCache[hit.ParentID]; exists {
				continue
			}
			var p ParentDoc
			var metadataJSON []byte
			err := pool.QueryRow(ctx, sqlParent, hit.ParentID).Scan(&p.ID, &p.Content, &metadataJSON)
			if err != nil {
				log.Printf("Error fetching parent %s: %v", hit.ParentID, err)
				continue
			}
			if err := json.Unmarshal(metadataJSON, &p.Metadata); err != nil {
				log.Printf("Error parsing parent metadata %s: %v", hit.ParentID, err)
				continue
			}
			parentCache[hit.ParentID] = &p
		}

		// D. Build output — parent content for LLM context, child metadata for deep links
		var finalOutput strings.Builder

		for _, hit := range hits {
			parent, ok := parentCache[hit.ParentID]
			if !ok {
				// Fallback: use the child's own content if parent fetch failed
				finalOutput.WriteString(fmt.Sprintf("---\n%s\n\n", hit.Content))
				continue
			}

			// Deep link from child metadata
			anchorBlockID, _ := hit.Metadata["anchor_block_id"].(string)
			pageURL, _ := hit.Metadata["url"].(string)
			title, _ := hit.Metadata["title"].(string)

			deepLink := pageURL
			if anchorBlockID != "" {
				cleanBlockID := strings.ReplaceAll(anchorBlockID, "-", "")
				// Build deep link: page URL base + #block anchor
				notionPageID, _ := hit.Metadata["notion_page_id"].(string)
				cleanPageID := strings.ReplaceAll(notionPageID, "-", "")
				deepLink = fmt.Sprintf("https://notion.so/%s#%s", cleanPageID, cleanBlockID)
			}

			finalOutput.WriteString(fmt.Sprintf("---\nTitle: %s\n%s\nSource: %s\n\n", title, parent.Content, deepLink))
		}

		return mcp.NewToolResultText(finalOutput.String()), nil
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

	if *transport == "stdio" {
		// Log to stderr in stdio mode to avoid corrupting JSON-RPC on stdout
		log.SetOutput(os.Stderr)
		fmt.Fprintln(os.Stderr, "MCP Server starting in stdio mode...")

		if err := server.ServeStdio(s); err != nil {
			fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
		}
		return
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
