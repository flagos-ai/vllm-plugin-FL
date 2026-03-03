# Claude Code Skills for vllm-plugin-FL

## Quick Start

```bash
# 在项目根目录下执行（二选一）
ln -s skills/claude .claude    # 推荐：创建软链接
# mv skills/claude .claude     # 或者：直接重命名
```

然后在 Claude Code 中即可使用：

```
/model-migrate-fl qwen3_5
```

## 可用 Skills

### model-migrate-fl

将最新 vLLM upstream 中的模型迁移到 vllm-plugin-FL（vLLM v0.13.0）。

```
/model-migrate-fl <model_name> [upstream_folder] [plugin_folder]
```

- `model_name` — 必填，snake_case 格式（如 `qwen3_5`, `kimi_k25`）
- `upstream_folder` — 可选，默认 `/tmp/vllm-upstream-ref`
- `plugin_folder` — 可选，默认当前目录

执行后 Claude 会自动完成：克隆上游代码 → 创建 config bridge → 复制模型文件并打补丁 → 注册 → 代码审查 → 单元测试 → 功能测试 → Benchmark → Serve 验证。

## 前置条件

- vLLM 0.13.0 已安装
- `pip install -e .` 安装了 vllm-plugin-FL
- Python 3.8+，GPU 环境

## 文件结构

```
skills/claude/
├── README.md                                    # 本文件
├── settings.local.json                          # Claude Code 权限配置
└── skills/model-migrate-fl/
    ├── SKILL.md                                 # Skill 入口
    ├── references/
    │   ├── procedure.md                         # 迁移步骤
    │   ├── compatibility-patches.md             # v0.13.0 补丁目录
    │   └── operational-rules.md                 # 运行规则
    └── scripts/
        ├── validate_migration.py                # 迁移代码自动审查
        ├── benchmark.sh                         # Benchmark 验证
        ├── serve.sh                             # Serve 启动
        └── request.sh                           # 请求测试
```
