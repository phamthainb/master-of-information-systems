#!/bin/bash

BRANCH="main"

echo "=== Reset local branch $BRANCH để khớp remote ==="
git fetch origin
git checkout $BRANCH
git reset --hard origin/$BRANCH
echo "=== Hoàn tất ==="
