#!/bin/bash

# 一键同步脚本 - 强制拉取最新代码（以repo为准）
# 用于将服务器代码更新到最新版本

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║         TWIST_CMG 代码同步脚本 (强制更新)               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# 检查是否在git仓库中
if [ ! -d ".git" ]; then
    echo "✗ 错误：当前目录不是git仓库"
    echo "  请在 TWIST_CMG 根目录运行此脚本"
    exit 1
fi

echo "当前目录: $(pwd)"
echo "当前分支: $(git branch --show-current)"
echo ""

# 显示本地修改
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "检查本地修改..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -n "$(git status --porcelain)" ]; then
    echo ""
    echo "⚠️  发现本地修改："
    echo ""
    git status --short
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚠️  警告：本地修改将被丢弃！"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    read -p "是否继续？所有本地修改将丢失 (yes/no): " -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        echo "❌ 已取消"
        exit 0
    fi
else
    echo "✓ 没有本地修改"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "开始同步..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 步骤1: 获取远程更新
echo "[1/4] 获取远程更新..."
git fetch origin

# 步骤2: 保存当前分支
current_branch=$(git branch --show-current)
echo "    当前分支: $current_branch"

# 步骤3: 重置所有本地修改
echo ""
echo "[2/4] 重置本地修改..."
echo "    - 清理未追踪文件..."
git clean -fd
echo "    - 丢弃所有本地修改..."
git reset --hard HEAD

# 步骤4: 强制拉取最新代码
echo ""
echo "[3/4] 拉取最新代码..."
git reset --hard origin/$current_branch

# 步骤5: 验证关键文件
echo ""
echo "[4/4] 验证关键文件..."

critical_files=(
    "pose/pose/utils/motion_lib_cmg.py"
    "legged_gym/legged_gym/envs/base/humanoid_mimic.py"
    "legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py"
    "test_cmg_integration.py"
)

all_ok=true
for file in "${critical_files[@]}"; do
    if [ -f "$file" ]; then
        echo "    ✓ $file"
    else
        echo "    ✗ $file (缺失)"
        all_ok=false
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$all_ok" = true ]; then
    echo "✅ 同步完成！"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "当前版本信息："
    echo "  提交: $(git rev-parse --short HEAD)"
    echo "  日期: $(git log -1 --format=%cd --date=format:'%Y-%m-%d %H:%M:%S')"
    echo "  作者: $(git log -1 --format=%an)"
    echo "  信息: $(git log -1 --format=%s)"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "下一步："
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  1. 测试集成:"
    echo "     python test_cmg_integration.py"
    echo ""
    echo "  2. 开始训练:"
    echo "     bash train_teacher.sh test_cmg cuda:0"
    echo ""
    
    # 检查CMG模型
    if [ ! -f "CMG_Ref/runs/cmg_20260123_194851/cmg_final.pt" ]; then
        echo "  ⚠️  提示: CMG模型文件不存在"
        echo "     需要先训练或复制模型文件到:"
        echo "     CMG_Ref/runs/cmg_20260123_194851/cmg_final.pt"
        echo ""
    fi
    
    exit 0
else
    echo "❌ 同步失败 - 部分关键文件缺失"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "可能的原因："
    echo "  1. Git仓库不完整"
    echo "  2. 文件未提交到仓库"
    echo "  3. .gitignore忽略了这些文件"
    echo ""
    echo "请检查git状态: git status"
    echo ""
    exit 1
fi
