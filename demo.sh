#!/bin/bash
set -e

echo "=== Semantic LLM Router Demo ==="
echo "Building Docker image..."
docker compose up --build -d

echo "Waiting for server..."
for i in $(seq 1 30); do
    if curl -s localhost:8000/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "Server failed to start"
        docker compose down
        exit 1
    fi
    sleep 2
done

QUERIES=(
    "hire a software engineer"
    "draft a non-disclosure agreement"
    "how do I become GDPR compliant"
    "write a cold email for enterprise sales"
    "design a microservices architecture"
    "what is the weather today"
    "terminate an employee for compliance violation"
)

for q in "${QUERIES[@]}"; do
    echo -e "\n--- Query: $q ---"
    curl -s -X POST localhost:8000/route \
        -H 'Content-Type: application/json' \
        -d "{\"query\":\"$q\"}" | python3 -m json.tool
done

echo -e "\n=== Demo Complete ==="
docker compose down
