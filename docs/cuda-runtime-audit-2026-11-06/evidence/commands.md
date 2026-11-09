# 复核命令（不含凭据）

在 H20 上准备独立目录与容器，镜像统一为：

```text
harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.3:2.3.0
```

当前仓库基线：`4b3a76a19de52af9981237e860c2177e73b9ee68`。

```bash
docker pull harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.3:2.3.0

# 将独立证据目录挂载为 /audit；下列为本次实际的目录与容器名。
docker run -d --name codex-cuda-runtime-trial-20261106 \
  --gpus all --shm-size 32g \
  -v /data/codex-cuda-runtime-audit-20261106:/audit \
  -v /data:/data -w /audit/repo \
  harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.3:2.3.0 \
  bash -lc 'sleep infinity'

docker exec codex-cuda-runtime-trial-20261106 bash -lc \
  'python tests/run.py --platform nvidia --device h20 --scope unit'
```

受保护约束 `protected.txt`：

```text
torch==2.13.0+cu130
torchvision==0.28.0+cu130
torchaudio==2.11.0+cu130
numpy==1.26.4
flag-gems==5.4.0
flagtree==0.7.0
flagcx==0.14.1+cuda13.3
```

失败的只解析命令：

```bash
docker exec codex-cuda-runtime-trial-20261106 bash -lc \
  'timeout 180 uv pip install --dry-run --python /flagos/bin/python \
     --index-url https://mirrors.aliyun.com/pypi/simple \
     --constraint /audit/protected.txt "vllm[audio]==0.32.0"'
```

成功解析的对照只从 protected.txt 中移除 NumPy 一行，其他约束相同：

```bash
grep -v '^numpy==' protected.txt > protected-without-numpy.txt
docker exec codex-cuda-runtime-trial-20261106 bash -lc \
  'timeout 180 uv pip install --dry-run --python /flagos/bin/python \
     --index-url https://mirrors.aliyun.com/pypi/simple \
     --constraint /audit/protected-without-numpy.txt "vllm[audio]==0.32.0"'
```

镜像站只改变包下载来源；本次没有执行安装。上述 stdout/stderr 分别见 uv-dry-run.log、uv-dry-run-allow-numpy.log。另一次默认索引 pip dry-run 获取元数据较慢，未用作成功安装证据。

FlagGems 核心检查在相同 Runtime 的默认编译器路径中执行：

```python
import torch, flag_gems
x = torch.randn(1024, device="cuda")
y = torch.randn_like(x)
expected = x + y
with flag_gems.use_gems():
    actual = x + y
torch.testing.assert_close(actual, expected)
print("FlagGems GPU add PASS")
```

基础清单生成脚本见 inventory.py。最终通过 importlib.metadata 再次读取当前解释器的完整包名/版本映射，与 packages.json 对比输出 `BASE_UNCHANGED True`。

清理状态：`docker stop codex-cuda-runtime-trial-20261106` 被堡垒机风险确认拦住，确认输入未能完成该操作；未尝试绕过。因此本次 trial 容器仍保留运行，原始命令为 sleep infinity，另有只解析依赖的 pip 任务可能尚在等待网络。不要将其计为已清理。基础 inventory 容器已正常退出。其他任务容器未停止。
