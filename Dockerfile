# Build Stage
FROM golang:1.25-alpine AS builder

WORKDIR /app

# Install git for fetching dependencies
RUN apk add --no-cache git

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download all dependencies. Dependencies will be cached if the go.mod and go.sum files are not changed
RUN go mod download

# Copy the source code
COPY . .

# Build the application
# CGO_ENABLED=0 creates a statically linked binary
RUN CGO_ENABLED=0 GOOS=linux go build -o mcp-server main.go

# Run Stage
FROM alpine:latest

WORKDIR /root/

# Copy the binary from the builder stage
COPY --from=builder /app/mcp-server .

# Install ca-certificates for HTTPS connections (needed for Supabase/OpenAI)
RUN apk --no-cache add ca-certificates

# Expose the port
EXPOSE 8080

# Command to run the executable
CMD ["./mcp-server"]
