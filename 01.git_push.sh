#!/bin/bash

BRANCH="main"
KEEP=3
MSG="${1:-Update}"

echo "=== Add, commit và push, chỉ giữ lại $KEEP commit cuối trên branch '$BRANCH' ==="

# Đảm bảo đang đúng branch
git checkout $BRANCH

# Add và commit nếu có thay đổi
git add .
git diff --cached --quiet
if [ $? -ne 0 ]; then
    git commit -m "$MSG"
    echo "✓ Commit thành công: $MSG"
else
    echo "✓ Không có thay đổi để commit."
fi

# Push tạm để đồng bộ với remote
git push origin $BRANCH

# Lấy số commit hiện tại
NUM_COMMITS=$(git rev-list --count HEAD)
if [ "$NUM_COMMITS" -le "$KEEP" ]; then
    echo "✓ Repo hiện tại chỉ có $NUM_COMMITS commit. Không cần trim."
    echo "✓ Kết thúc."
    exit 0
fi

# Tìm commit cũ nhất cần giữ
OLDEST=$(git rev-list --max-count=$KEEP HEAD | tail -n1)

echo "OLDEST: $OLDEST"

if [ -z "$OLDEST" ]; then
    echo "✗ Không tìm thấy commit cũ nhất. Bỏ qua bước trim."
    exit 0
fi


echo "✓ Trim repo: giữ commit từ $OLDEST đến HEAD"


git checkout --detach $OLDEST
git checkout -b temp

git cherry-pick $(git rev-list --reverse $OLDEST^..main)


git push origin temp:main --force

git checkout main
git reset --hard origin/main
git branch -D temp


