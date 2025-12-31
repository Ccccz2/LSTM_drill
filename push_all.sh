#!/bin/bash

# 自动上传脚本

# 暂存所有文件（排除 .gitignore 忽略的）
git add .

# 提交到本地仓库
# 如果提供了参数，就用参数作为提交信息；否则默认
if [ -z "$1" ]; then
    git commit -m "Auto update all files"
else
    git commit -m "$1"
fi

# 推送到 GitHub
git push

echo " All changes have been pushed to GitHub."