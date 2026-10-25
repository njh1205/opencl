# 协作规范（Contributing）— fork + PR 模式

本仓库：`njh1205/opencl`（**Public**）— hpocl OpenCL kernel 库，目标平台 QX_49DR_SV3 (XCZU49DR)。

本仓库是**公开**的，任何人都可以 clone / fork / 提 Pull Request，**无需任何邀请**。
main 分支受保护：合并必须走 Pull Request，由 owner review 后批准。

## 任何人都能参与（不需要被邀请）

- 直接浏览/克隆：`git clone git@github.com:njh1205/opencl.git`
- 想改代码贡献：**fork** 到自己的账号 → 在**自己的 fork** 里随便改（无需任何许可）→ 提 PR 回本仓库 → owner 批准后合并。

## Fork + PR 全流程

```bash
# 1. 网页上 Fork：https://github.com/njh1205/opencl 右上角 Fork（fork 到你自己账号）

# 2. clone 你自己 的 fork（不是原仓库）
git clone git@github.com:<你的用户名>/opencl.git
cd opencl

# 3. 关联原仓库为 upstream，并同步最新 main
git remote add upstream git@github.com:njh1205/opencl.git
git fetch upstream
git checkout -b feature/你的功能名 upstream/main

# 4. 改代码，本地提交
git add -A
git commit -m "简洁说明改了什么、为什么"

# 5. push 到【你自己的 fork】（push 自己的 fork 无需任何许可）
git push -u origin feature/你的功能名

# 6. 网页上提 Pull Request
#    GitHub 会提示 "Compare & pull request" → base: njh1205/opencl main ← 你的分支
#    写清改动说明与测试结果 → Create pull request
#    等 owner review；合并后可以删掉自己的远端分支：
git push origin --delete feature/你的功能名
```

## 长期保持 fork 同步（每次开工前）

```bash
git fetch upstream
git checkout main
git merge upstream/main        # 或 git rebase upstream/main
git push origin main           # 同步到你自己的 fork
```

## 硬性禁止（Public 仓库，尤其重要）

- ❌ **提交任何敏感信息**——本仓库公开，**提交历史永久公开**（密钥、密码、token、内网 IP、私人路径、未公开论文手稿都严禁 `git add`）
- ❌ 提交编译产物：`.gitignore` 已排除 `/build/`，**不要 `git add -f` 强行加进去**
- ❌ `git push --force` 到已开 PR 的分支（会破坏 review 历史；确需更新分支时用 `git push --force-with-lease`）
- ❌ 直接 push 到 `njh1205/opencl`（你也没有这个权限；你的 push 目标始终是**自己的 fork**）

## 构建/测试备忘

- 本机 Mac (Apple GPU)：`make check` 全套 bench（op/cf/decomp/datapath/chain/rx/pipeline/gh/tf）。
- 测试向量在 `build/` 下由 `make` 生成，不入库；新机器先 `make` 再跑 bench。
- 提交前至少跑一遍相关 bench：例如 `./build/op_bench` 覆盖 kernel 对拍。
