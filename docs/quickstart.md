# 快速开始

这是从研究工作区整理的独立发布目录，不需要同级 RetrievalLab。运行时使用随仓库保留的 503 个检索块和表格引用注册表；模型与向量缓存在本机准备，不提交 Git。

## 1. 安装

以下命令均在仓库根目录执行。建议 Python 3.12，准备 uv；在线问答还需要 Docker Compose 和 DeepSeek API Key。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.rag.lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
```

已有虚拟环境时跳过创建。依赖锁来自开发环境；本次发布整理的验证范围见下文，尚未完成另一台机器的全新依赖安装验证。

## 2. 不调用模型服务的检查

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/demo_parse_chunk.py
```

测试使用模型/检索替身和本地资产，不需要 API Key、ES 或下载 BGE。示例入口运行真实 HTML 适配器和三种切分器，输出到 `outputs/parse_chunk_demo/`。示例文字是合成的，用于展示处理流程，不是官方产品事实。

## 3. 准备本地向量

```bash
.venv/bin/python scripts/prepare_vectors.py
```

首次运行按固定 revision 下载 BGE 模型至 `.cache/models/`，对随附 chunk 编码，拒绝静默截断，并生成：

- `.cache/document_vectors.npz`：本机文档向量。
- `config/rag.local.json`：实际向量哈希及独立 demo 索引名。

已有上述输出时不覆盖，可直接进入下一步。机器、依赖或计算环境变化可能造成向量哈希不同；本地配置记录真实结果，不冒充历史评测快照。`config/rag.v1.json` 保留导出时的参数模板与历史向量哈希。

## 4. 配置服务与索引

首次使用时将 `.env.example` 复制为 `.env`，填入 `DEEPSEEK_API_KEY`。保留示例中的 `ES_URL=http://127.0.0.1:19201`，与此目录的 Compose 一致；已有环境变量优先于 `.env`。

```bash
docker compose up -d elasticsearch
docker compose ps
```

等待服务状态为 healthy。此 Compose 使用本地端口 19201、独立数据卷，避免与原实验服务混用。

首次创建索引：

```bash
.venv/bin/python scripts/prepare_rag_index.py --config config/rag.local.json
```

已有索引时改用校验命令：

```bash
.venv/bin/python scripts/prepare_rag_index.py --config config/rag.local.json --verify-only
```

建索引入口拒绝覆盖已有索引，导入后写保护。索引的文本、向量与来源配置必须一致。发布版的建索引和聊天入口均读取 `.env`；直接从 Python 使用运行时类时，需自行加载配置。

## 5. 启动问答

```bash
.venv/bin/python scripts/chat_rag.py --config config/rag.local.json --question "DJI RC 2 的屏幕多大？它自己有存储空间吗？"
.venv/bin/python scripts/chat_rag.py --config config/rag.local.json --thread demo --output outputs/chat.jsonl
```

第二条为多轮交互：`/new 问题` 换话题，`/quit` 退出。会话仅在当前进程保存。JSONL 包含用户输入与运行 trace，位于 Git 忽略的 `outputs/`。

## 发布目录的范围

`src/miniassist/` 保留解析、切分与问答核心代码；四个脚本分别负责解析演示、向量准备、索引与聊天。原始 PDF/网页归档、人工金标、历次实验日志和一次性修复脚本仍在研究工程，本目录不包含完整历史实验重放链。原始资料的批量重解析模块需要对应的源文件与 canonical 文本，演示入口使用随附合成输入。

评估数字是历史冻结结果，不是本地重建索引后自动获得的成绩。详细范围见 [评估说明](evaluation.md)，代码与必要资产哈希见 [发布清单](release_manifest.json)。

本次独立目录的实际检查见 [发布验证](verification.md)。
