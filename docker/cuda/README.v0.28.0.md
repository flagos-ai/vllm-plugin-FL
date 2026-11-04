# NVIDIA CUDA 13.3 / vLLM 0.28.0 CI 镜像

已发布镜像：`harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.28.0-cuda-ci`

镜像摘要：`sha256:96f89fec934a02a59b12cc1cb88c8372249d8b615039794bcefdb40fb04d5a3a`

`Dockerfile.v0.28.0` 保存本次镜像的构建配方，基础镜像通过摘要固定到
FlagOS runtime NVIDIA CUDA 13.3 的 2.3.0 版本。保留系统 CUDA 13.3，补齐
13.3 编译工具及开发库，安装 vLLM 0.28.0、PyTorch cu130、FlagTree 和固定提交的
FlagGems、vllm-plugin-FL。Python 依赖约束见 `constraints.v0.28.0.txt`。

## 构建

在仓库根目录执行，构建上下文必须为 `docker/cuda/`：

```bash
docker build \
  -f docker/cuda/Dockerfile.v0.28.0 \
  -t harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.28.0-cuda-ci \
  docker/cuda/
```

该版本是单阶段 CI 配方，无需指定 `--target`。
`docker/build.sh` 仍使用默认的 `docker/cuda/Dockerfile`；构建此版本请使用上述命令。
构建需要访问基础镜像仓库、系统及 Python 软件源、GitHub 和 vLLM 音频资源。
本次仅将已构建配方纳入仓库，未重新构建镜像。
