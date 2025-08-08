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

if [ -z "$OLDEST" ]; then
  echo "✗ Không tìm thấy commit cũ nhất. Bỏ qua bước trim."
  exit 0
fi

# Nếu HEAD == OLDEST thì chỉ có đúng 1 commit cần giữ
if [ "$OLDEST" == "$(git rev-parse HEAD)" ]; then
  echo "✓ HEAD chính là commit duy nhất cần giữ. Không cần cherry-pick."
  echo "✓ Kết thúc."
  exit 0
fi

echo "✓ Trim repo: giữ commit từ $OLDEST đến HEAD"

# Tạo branch orphan từ commit cũ nhất
git checkout --orphan temp $OLDEST
if [ $? -ne 0 ]; then
  echo "✗ Không thể tạo orphan branch."
  git checkout $BRANCH
  exit 0
fi

# Cherry-pick các commit cần giữ
git cherry-pick $OLDEST..HEAD
if [ $? -ne 0 ]; then
  echo "✗ Cherry-pick thất bại. Hủy thao tác."
  git checkout $BRANCH
  git branch -D temp
  exit 0
fi

# Xoá branch cũ và đổi tên lại
git branch -D $BRANCH
git branch -m $BRANCH

# Force push lên remote
git push origin $BRANCH --force

echo "✓ Hoàn tất! Repo giờ chỉ còn $KEEP commit cuối cùng."
