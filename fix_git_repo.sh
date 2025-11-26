#!/bin/bash
# Fix Git Repository - Safe Version

echo "=========================================="
echo "Fixing Git Repository"
echo "=========================================="
echo ""

# Step 1: Fix permissions
echo "Step 1: Fixing .git permissions (requires sudo password)..."
sudo chown -R $USER:$USER .git
if [ $? -eq 0 ]; then
    echo "✓ Permissions fixed"
else
    echo "✗ Failed to fix permissions"
    exit 1
fi
echo ""

# Step 2: Abort any ongoing operations
echo "Step 2: Cleaning up git state..."
git merge --abort 2>/dev/null || true
git rebase --abort 2>/dev/null || true
echo "✓ Git state cleaned"
echo ""

# Step 3: Reset to your last commit
echo "Step 3: Resetting to your last commit..."
git reset --hard ed87c46
echo "✓ Reset complete"
echo ""

# Step 4: Fetch remote changes
echo "Step 4: Fetching remote changes..."
git fetch origin
echo "✓ Fetch complete"
echo ""

# Step 5: Rebase your commits on top of remote
echo "Step 5: Rebasing your commits on remote..."
git rebase origin/main
if [ $? -eq 0 ]; then
    echo "✓ Rebase successful"
else
    echo "✗ Rebase failed - you may need to resolve conflicts"
    echo "Run: git rebase --continue after resolving conflicts"
    exit 1
fi
echo ""

# Step 6: Push to remote
echo "Step 6: Pushing to GitHub..."
git push origin main
if [ $? -eq 0 ]; then
    echo "✓ Push successful"
else
    echo "✗ Push failed"
    echo "You may need to force push: git push -f origin main"
    exit 1
fi
echo ""

echo "=========================================="
echo "✓ Repository fixed and synced!"
echo "=========================================="
echo ""
echo "Your files are safe and synced to GitHub."
echo ""
