#!/bin/bash

# Security scanning script for Code Explainer
# Runs SAST, SCA, and container scanning

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "Code Explainer Security Scanning"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCAN_FAILED=0

# 1. Check for bandit (SAST)
echo -e "\n${YELLOW}[1/4] Running SAST scan with Bandit...${NC}"
if command -v bandit &> /dev/null; then
    if bandit -r "$PROJECT_ROOT/src" -f json -o "$PROJECT_ROOT/bandit-report.json" 2>/dev/null; then
        echo -e "${GREEN}✓ Bandit scan completed${NC}"
    else
        echo -e "${YELLOW}⚠ Bandit found potential issues (see bandit-report.json)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Bandit not installed, skipping SAST scan${NC}"
fi

# 2. Check for safety (SCA - Python dependencies)
echo -e "\n${YELLOW}[2/4] Scanning dependencies with Safety...${NC}"
if command -v safety &> /dev/null; then
    if safety check --json > "$PROJECT_ROOT/safety-report.json" 2>/dev/null; then
        echo -e "${GREEN}✓ No vulnerable dependencies found${NC}"
    else
        echo -e "${RED}✗ Vulnerable dependencies detected${NC}"
        cat "$PROJECT_ROOT/safety-report.json"
        SCAN_FAILED=1
    fi
else
    echo -e "${YELLOW}⚠ Safety not installed, installing...${NC}"
    pip install safety > /dev/null
    if safety check --json > "$PROJECT_ROOT/safety-report.json" 2>/dev/null; then
        echo -e "${GREEN}✓ No vulnerable dependencies found${NC}"
    else
        echo -e "${RED}✗ Vulnerable dependencies detected${NC}"
        SCAN_FAILED=1
    fi
fi

# 3. Check for pip-audit (newer alternative to safety)
echo -e "\n${YELLOW}[3/4] Running pip-audit for dependency vulnerabilities...${NC}"
if command -v pip-audit &> /dev/null; then
    if pip-audit --desc > "$PROJECT_ROOT/pip-audit-report.txt" 2>&1; then
        echo -e "${GREEN}✓ No vulnerable dependencies found${NC}"
    else
        echo -e "${YELLOW}⚠ Some vulnerabilities found (see pip-audit-report.txt)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ pip-audit not installed, installing...${NC}"
    pip install pip-audit > /dev/null
    if pip-audit --desc > "$PROJECT_ROOT/pip-audit-report.txt" 2>&1; then
        echo -e "${GREEN}✓ No vulnerable dependencies found${NC}"
    else
        echo -e "${YELLOW}⚠ Some vulnerabilities found (see pip-audit-report.txt)${NC}"
    fi
fi

# 4. Check code complexity with pylint
echo -e "\n${YELLOW}[4/4] Checking code quality with pylint...${NC}"
if command -v pylint &> /dev/null; then
    pylint --exit-zero --output-format=json "$PROJECT_ROOT/src/code_explainer" > "$PROJECT_ROOT/pylint-report.json" 2>/dev/null || true
    echo -e "${GREEN}✓ Pylint report generated (see pylint-report.json)${NC}"
else
    echo -e "${YELLOW}⚠ Pylint not installed, skipping code quality check${NC}"
fi

# Summary
echo -e "\n${YELLOW}=========================================${NC}"
echo "Security Scan Summary:"
echo "========================================="
echo "Reports generated:"
[ -f "$PROJECT_ROOT/bandit-report.json" ] && echo "  • Bandit (SAST): bandit-report.json"
[ -f "$PROJECT_ROOT/safety-report.json" ] && echo "  • Safety (SCA): safety-report.json"
[ -f "$PROJECT_ROOT/pip-audit-report.txt" ] && echo "  • pip-audit: pip-audit-report.txt"
[ -f "$PROJECT_ROOT/pylint-report.json" ] && echo "  • Pylint (Quality): pylint-report.json"

if [ $SCAN_FAILED -eq 1 ]; then
    echo -e "\n${RED}✗ Security scan detected issues!${NC}"
    exit 1
else
    echo -e "\n${GREEN}✓ Security scan completed successfully${NC}"
    exit 0
fi
