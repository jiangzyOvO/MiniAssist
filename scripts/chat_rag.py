"""CLI chat. Checkpoints live for this process; JSONL exports never contain gold answers."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import replace
from miniassist.rag.graph import RAGService
from miniassist.rag.llm import DeepSeek
from miniassist.rag.retriever import HybridRetriever
from miniassist.rag.settings import Settings, load_config, load_local_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default="local")
    parser.add_argument("--question", help="Single turn; omit for multi-turn chat")
    parser.add_argument("--output", type=Path, help="Optional local JSONL trace; includes user conversation")
    parser.add_argument("--config")
    parser.add_argument("--policy", choices=['B', 'B0', 'C', 'C_no_postcheck'], default='B')
    args = parser.parse_args()
    load_local_env()
    settings, config = Settings.from_env(), load_config(args.config)
    settings = replace(settings, policy=args.policy, surface_verification=args.policy == 'B')
    llm = DeepSeek(settings)  # Check key before loading embeddings.
    retriever = HybridRetriever(config, settings.es_url)
    service = RAGService(llm, retriever, settings)
    if not args.question:
        print("输入问题；/new 问题 开始新话题；/quit 退出。会话状态只保存在本进程。")
    while True:
        try:
            message = args.question or input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message == "/quit":
            break
        if not message:
            continue
        new_topic = message.startswith("/new ")
        state = service.ask(args.thread, message[5:] if new_topic else message, new_topic=new_topic)
        print(f"\n[{state['status']}] {state['response']}\n")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            record = {"timestamp": datetime.now(timezone.utc).isoformat(), "thread_id": args.thread,
                      "model": settings.model, "retrieval_config": {k: v for k, v in config.items()
                       if k not in {"vectors", "model_cache"}}, "state": state,
                      "question": state["original_question"], "resolved_question": state["question"],
                      "response": state["response"],
                      "retrieved_contexts": [{"doc_id": c["chunk_id"], "text": c["text"]}
                                             for c in state["evidence"]]}
            with args.output.open("a") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        if args.question:
            break


if __name__ == "__main__":
    main()
