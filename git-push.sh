#!/bin/bash

# SecureView Alert - Git Push Script
# Complete deployment commands

echo "========================================"
echo "SecureView Alert - Git Push Script"
echo "========================================"
echo ""

# Step 1: Check git status
echo "📊 Checking git status..."
git status
echo ""

# Step 2: Add all changes
echo "➕ Adding all changes..."
git add -A
echo "✅ All files staged"
echo ""

# Step 3: Show what will be committed
echo "📝 Changes to commit:"
git status --short
echo ""

# Step 4: Commit
echo "💾 Creating commit..."
git commit -m "SecureView Alert: Production Release

Features:
- Fixed React Live Monitor integration
- Backend fallback mode (stable on macOS)
- MJPEG video streaming
- Real-time weapon/people detection
- MongoDB integration
- React dashboard with all routes
- White theme UI
- Comprehensive error handling

Changes:
- Updated backend/server.py with fallback mode
- Fixed frontend/.env for 127.0.0.1 connection
- Updated App.js routing for Live Monitor
- Added LiveMonitor.js and CSS
- Cleaned up unnecessary files
- Updated .gitignore
- Added comprehensive README

Status: Production Ready ✅"

echo ""
echo "✅ Commit created"
echo ""

# Step 5: Show commit log
echo "📋 Latest commits:"
git log --oneline -5
echo ""

# Step 6: Push to main
echo "🚀 Ready to push to main branch!"
echo ""
echo "Run these commands to push:"
echo ""
echo "  git push origin main"
echo ""
echo "Or with force (if needed):"
echo "  git push -u origin main"
echo ""
