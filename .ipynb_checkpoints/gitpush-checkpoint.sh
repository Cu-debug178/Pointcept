#!/bin/bash

# 检查是否在 Git 仓库内
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "❌ 错误：当前目录不是 Git 仓库"
    exit 1
fi

# 显示当前状态
echo "📁 当前仓库：$(basename $(git rev-parse --show-toplevel))"
echo "🌿 当前分支：$(git branch --show-current)"
echo ""
git status -s

# 如果没有更改，直接退出
if [ -z "$(git status -s)" ]; then
    echo "✅ 没有需要提交的更改"
    exit 0
fi

# 询问提交信息
echo ""
read -p "✏️ 请输入提交信息（直接回车使用默认时间）: " msg
if [ -z "$msg" ]; then
    msg="Update $(date +'%Y-%m-%d %H:%M:%S')"
fi

# 添加、提交、推送
git add -A
git commit -m "$msg"
echo ""
echo "🚀 正在推送到远程..."
git push

if [ $? -eq 0 ]; then
    echo "✅ 提交并推送成功！"
else
    echo "❌ 推送失败，请检查网络或远程仓库状态"
fi