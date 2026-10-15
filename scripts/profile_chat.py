"""Measure chat latency through the API: time to first token and total time.

Usage: python scripts/profile_chat.py --mode stream   (or --mode chat)
Needs only `requests`. Questions run in one session so the follow-up has context.
"""

import argparse
import json
import time
import uuid

import requests

QUESTIONS = [
    ("chào hỏi", "xin chào"),
    ("ngoài lề", "1 + 1 = ?"),
    ("pháp luật", "Đi xe máy không đội mũ bảo hiểm bị phạt bao nhiêu?"),
    ("tiếp nối", "Thế còn xe đạp điện thì sao?"),
]


def run_chat(url: str, session_id: str, message: str) -> tuple[float | None, float, str]:
    """POST /chat: nothing is visible until the whole answer is ready."""
    t0 = time.perf_counter()
    r = requests.post(f"{url}/chat", json={"session_id": session_id, "message": message}, timeout=180)
    total = time.perf_counter() - t0
    r.raise_for_status()
    return total, total, r.json()["answer"]


def run_stream(url: str, session_id: str, message: str) -> tuple[float | None, float, str]:
    """POST /chat/stream: the first token is what the user starts reading."""
    t0 = time.perf_counter()
    first = None
    answer = ""
    with requests.post(
        f"{url}/chat/stream", json={"session_id": session_id, "message": message}, timeout=180, stream=True
    ) as r:
        r.raise_for_status()
        r.encoding = "utf-8"
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            event = json.loads(line)
            if event["type"] == "token":
                if first is None:
                    first = time.perf_counter() - t0
                answer += event["text"]
            elif event["type"] == "error":
                raise RuntimeError(event["detail"])
    return first, time.perf_counter() - t0, answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure chat latency through the API.")
    parser.add_argument("--mode", choices=["chat", "stream"], required=True)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--pause", type=float, default=5.0, help="Seconds between questions (rate limits).")
    args = parser.parse_args()

    run = run_chat if args.mode == "chat" else run_stream
    session_id = f"profile-{uuid.uuid4()}"
    print(f"| Loại câu | Câu hỏi | Token đầu tiên | Tổng | Số ký tự trả lời |\n|---|---|---|---|---|")
    for kind, question in QUESTIONS:
        first, total, answer = run(args.url, session_id, question)
        print(f"| {kind} | {question} | {first:.1f}s | {total:.1f}s | {len(answer)} |", flush=True)
        time.sleep(args.pause)
    requests.delete(f"{args.url}/chat/{session_id}", timeout=10)


if __name__ == "__main__":
    main()
