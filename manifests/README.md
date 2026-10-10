# 产物清单

`build/`、`runs/`、`baseline_preds/`、`local_exports/` 被 gitignore，只有本地一份。这里记录它们的 sha256，用于发现丢失或损坏。

- `<dir>.sha256`：普通文件，`sha256sum` 格式，路径相对仓库根。`build/keeper/`（Colab 运维状态，会变）不收录。
- `symlinks.txt`：符号链接及其目标，不计算哈希。

校验（在仓库根执行）：

```bash
sha256sum --quiet -c manifests/runs.sha256
```

重新生成（以 runs 为例）：

```bash
find runs -type f -print0 | sort -z | xargs -0 -P 6 -n 8 sha256sum | sort -k2 > manifests/runs.sha256
```

生成日期：2026-10-10。
