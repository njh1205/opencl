# 协作规范（Contributing）

本仓库：`njh1205/opencl`（Private）— hpocl OpenCL kernel 库，目标平台 QX_49DR_SV3 (XCZU49DR)。

## 权限模型（先看这个）

- **`main` 分支受保护**：任何人都**不能直接 push 到 main**（含 owner 之外的所有协作者）。
- 改动进入 main 的唯一方式：**开分支 → 提 Pull Request → owner review 同意 → 合并**。
- 协作者（Write 权限）：只能 push 到自己创建的分支。

## 开发流程（每次改动）

```bash
# 1. 同步 + 开自己的分支（永远不要直接在 main 上改）
git fetch origin
git checkout -b feature/你的功能名 origin/main

# 2. 改代码，本地提交
git add -A
git commit -m "简洁说明改了什么、为什么"

# 3. push 你的分支
git push -u origin feature/你的功能名

# 4. 网页上提 Pull Request
#    GitHub 会显示 "Compare & pull request" 按钮 → 点它 →
#    base: main  ← 你的分支
#    写清改动说明、测试结果 → Create pull request
#    等 owner review + merge，合并后删除自己的远端分支：
git push origin --delete feature/你的功能名
```

## 硬性禁止

- ❌ push 到 `main`（会被保护规则拒绝）
- ❌ `git push --force` 到任何已存在分支
- ❌ 提交编译产物：`.gitignore` 已排除 `/build/`，**不要 `git add -f` 强行加进去**
- ❌ 改别人正在用的分支；一个分支只做一件事

## 构建/测试备忘

- 本机 Mac (Apple GPU)：`make check` 全套 bench（op/cf/decomp/datapath/chain/rx/pipeline/gh/tf）。
- 测试向量在 `build/` 下由 `make` 生成，不入库；新机器先 `make` 再跑 bench。
- 提交前至少跑一遍相关 bench：例如 `./build/op_bench` 覆盖 kernel 对拍。
