#!/bin/bash
# Health check script for Code Explainer services

set -e

echo "🏥 Code Explainer Health Check"
echo "================================"

# Check API health
echo -n "API Health Check... "
if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ OK"
    curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || echo ""
else
    echo "❌ FAILED"
    exit 1
fi

# Check API version
echo -n "API Version... "
if curl -sf http://localhost:8000/version > /dev/null 2>&1; then
    VERSION=$(curl -s http://localhost:8000/version | python3 -c "import sys,json; print(json.load(sys.stdin)['code_explainer_version'])" 2>/dev/null || echo "unknown")
    echo "✅ v$VERSION"
else
    echo "❌ FAILED"
fi

# Check metrics endpoint
echo -n "Metrics Endpoint... "
if curl -sf http://localhost:8000/metrics > /dev/null 2>&1; then
    echo "✅ OK"
else
    echo "⚠️  Not available (may require API key)"
fi

# Test explanation endpoint
echo -n "Explanation Endpoint... "
RESPONSE=$(curl -sf -X POST http://localhost:8000/explain \
    -H "Content-Type: application/json" \
    -d '{"code": "print(\"test\")"}' 2>/dev/null)

if [ $? -eq 0 ]; then
    echo "✅ OK"
else
    echo "❌ FAILED"
    exit 1
fi

echo ""
echo "✅ All health checks passed!"
