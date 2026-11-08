"""Does the local judge (vstash.judge) stop made-up answers without blocking real ones?

Ground truth from SQuAD 2.0 (validation, CC BY-SA 4.0): every Wikipedia article of the split is ingested
into a fresh vstash memory, one document per article. Half of the sampled questions are answerable from
it; the other half are SQuAD's *unanswerable* questions, written by crowdworkers to look answerable from
the very same paragraphs. Labels are human, so no model grades its own homework on answerability.

Three measurements:
  1. Gate alone (no LLM): P(answerable) for every question -> AUROC, and at each threshold the share of
     unanswerable questions blocked vs answerable questions wrongly blocked.
  2. End to end: ``ask`` without the judge vs with it, the LLM's reply graded ABSTAIN / CORRECT / WRONG
     against SQuAD's gold answers (a one-word grading call to the same LLM backend).
  3. Verify: how often claims get flagged in correct answers vs in wrong or invented ones.

    OPENAI_API_KEY=... python -m experiments.judge_squad --questions 200 \
        --base-url https://api.deepseek.com --model deepseek-flash

Every question (chunks, reply, grade) is kept in ``RECORDS``, so another judge can re-verify the very
same replies without new LLM calls (replies are not deterministic run to run):

    python -m experiments.judge_squad --reverify --verify-adapter ~/code/destila/adapters/jev \
        --out experiments/results/judge_squad_jev.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path("experiments/data/squad_v2")
OUT = Path("experiments/results/judge_squad.json")
RECORDS = DATA / "records.jsonl"
THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
GRADE = """Grade an assistant's RESPONSE to a QUESTION about some documents.
GOLD: {gold}

Reply with one word:
ABSTAIN - the response says the documents do not contain the answer (or declines to answer)
CORRECT - the response gives an answer that matches a GOLD answer
WRONG - the response gives an answer that does not match GOLD (when GOLD is "none", any concrete answer is WRONG)

QUESTION: {q}
RESPONSE: {r}"""


def load_squad() -> list[dict]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        "rajpurkar/squad_v2",
        "squad_v2/validation-00000-of-00001.parquet",
        repo_type="dataset",
        local_dir=DATA,
    )
    return pq.read_table(path).to_pylist()


def build_memory(rows: list[dict], db: Path):
    from vstash import Memory

    if db.exists():
        db.unlink()
    mem = Memory(db=db)
    articles: dict[str, list[str]] = {}
    for r in rows:
        paras = articles.setdefault(r["title"], [])
        if r["context"] not in paras:
            paras.append(r["context"])
    t = time.time()
    for title, paras in articles.items():
        mem.remember("\n\n".join(paras), title=title.replace("_", " "))
    print(
        f"memoria: {len(articles)} artículos, {sum(map(len, articles.values()))} párrafos, {time.time() - t:.0f}s"
    )
    return mem


def sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    rnd = random.Random(seed)
    ans = [r for r in rows if len(r["answers"]["text"])]
    imp = [r for r in rows if not len(r["answers"]["text"])]
    picked = rnd.sample(ans, n // 2) + rnd.sample(imp, n // 2)
    rnd.shuffle(picked)
    return [
        {
            "id": r["id"],
            "q": r["question"],
            "gold": sorted(set(r["answers"]["text"])),
            "title": r["title"],
        }
        for r in picked
    ]


def auroc(pos: list[float], neg: list[float]) -> float:
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / max(len(pos) * len(neg), 1)


def grade(client, model: str, q: dict, response: str, extra: dict | None = None) -> str:
    gold = " | ".join(q["gold"]) if q["gold"] else "none (the documents do not contain the answer)"
    msg = GRADE.format(gold=gold, q=q["q"], r=response[:4000])
    out = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": msg}],
        max_tokens=3,
        temperature=0,
        extra_body=extra,
    )
    word = (out.choices[0].message.content or "").strip().upper()
    return next((w for w in ("ABSTAIN", "CORRECT", "WRONG") if word.startswith(w[:3])), "WRONG")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--judge-model", default="LiquidAI/d1-2B")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verify-adapter", help="LoRA fact-checking adapter for the claim checks")
    ap.add_argument("--reverify", action="store_true", help="re-judge the saved replies only")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    if a.reverify:
        return reverify(a)

    from openai import OpenAI

    from vstash import judge as J
    from vstash.chat import ask_full
    from vstash.config import JudgeConfig, VstashConfig
    from vstash.models import Judgment

    rows = load_squad()
    mem = build_memory(rows, DATA / "memory.db")
    qs = sample(rows, a.questions, a.seed)
    extra = {"thinking": {"type": "disabled"}} if "deepseek" in a.base_url else None
    cfg = VstashConfig.model_validate(
        {
            "inference": {"backend": "openai"},
            "openai": {"model": a.model, "base_url": a.base_url, "extra_body": extra},
        }
    )
    jcfg = JudgeConfig(enabled=True, model=a.judge_model, verify_adapter=a.verify_adapter)
    judge = J.get_judge(jcfg)
    client = OpenAI(base_url=a.base_url)

    # 1. Gate: retrieval + one judge read per question, no LLM.
    t = time.time()
    for q in qs:
        q["chunks"] = mem.search(q["q"], top_k=a.top_k)
        q["retrieved_gold"] = bool(q["gold"]) and any(
            g.lower() in c.text.lower() for g in q["gold"] for c in q["chunks"]
        )
        q["p_answerable"] = J.gate(judge, q["q"], q["chunks"], jcfg).answerable
    gate_ms = (time.time() - t) * 1000 / len(qs)
    print(f"compuerta: {len(qs)} preguntas, {gate_ms:.0f} ms por pregunta (búsqueda + juez)")

    # 2. End to end: the LLM answers every question (the no-judge baseline); grading is independent of the judge.
    def answer(q: dict) -> None:
        q["response"] = ask_full(q["q"], q["chunks"], cfg).content
        q["grade"] = grade(client, a.model, q, q["response"], extra)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(answer, qs))

    # 3. Verify each reply's claims against its context.
    for q in qs:
        jm = J.verify(
            judge, Judgment(answerable=q["p_answerable"]), q["response"], q["chunks"], jcfg
        )
        q["claims"] = [{"text": c.text, "supported": c.supported} for c in jm.claims]

    with RECORDS.open("w") as f:
        for q in qs:
            rec = {k: v for k, v in q.items() if k not in ("chunks", "claims")}
            rec["chunks"] = [{"title": c.title, "text": c.text} for c in q["chunks"]]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    report(qs, a, gate_ms)


def reverify(a) -> None:
    """Same questions, same retrieved chunks, same replies: only the judge changes."""
    from types import SimpleNamespace

    from vstash import judge as J
    from vstash.config import JudgeConfig
    from vstash.models import Judgment

    qs = [json.loads(line) for line in RECORDS.open()]
    jcfg = JudgeConfig(enabled=True, model=a.judge_model, verify_adapter=a.verify_adapter)
    judge = J.get_judge(jcfg)
    t = time.time()
    for q in qs:
        chunks = [SimpleNamespace(**c) for c in q["chunks"]]
        jm = J.verify(judge, Judgment(answerable=q["p_answerable"]), q["response"], chunks, jcfg)
        q["claims"] = [{"text": c.text, "supported": c.supported} for c in jm.claims]
    print(f"verificación: {(time.time() - t) * 1000 / len(qs):.0f} ms por respuesta")
    report(qs, a, 0.0)


def report(qs: list[dict], a, gate_ms: float) -> None:
    ans = [q for q in qs if q["gold"]]
    imp = [q for q in qs if not q["gold"]]
    out: dict = {
        "setup": {
            "dataset": "SQuAD 2.0 validation (all 35 articles ingested)",
            "questions": len(qs),
            "answerable": len(ans),
            "unanswerable": len(imp),
            "llm": a.model,
            "judge": a.judge_model,
            "verify_adapter": a.verify_adapter,
            "top_k": a.top_k,
            "gate_ms_per_question": round(gate_ms),
        },
        "retrieval_hit_on_answerable": sum(q["retrieved_gold"] for q in ans) / len(ans),
        "gate_auroc": auroc([q["p_answerable"] for q in ans], [q["p_answerable"] for q in imp]),
        "gate_auroc_retrieved_only": auroc(
            [q["p_answerable"] for q in ans if q["retrieved_gold"]],
            [q["p_answerable"] for q in imp],
        ),
        "thresholds": {},
    }
    for th in THRESHOLDS:
        blocked_imp = sum(q["p_answerable"] < th for q in imp) / len(imp)
        blocked_ans = sum(q["p_answerable"] < th for q in ans) / len(ans)

        # With the judge: a gated question is an abstention; otherwise the LLM's own graded reply stands.
        def with_judge(group: list[dict], th: float = th) -> dict:
            g = ["ABSTAIN" if q["p_answerable"] < th else q["grade"] for q in group]
            return {k: g.count(k) / len(g) for k in ("CORRECT", "WRONG", "ABSTAIN")}

        out["thresholds"][str(th)] = {
            "blocks_unanswerable": blocked_imp,
            "blocks_answerable": blocked_ans,
            "answerable": with_judge(ans),
            "unanswerable": with_judge(imp),
        }

    def base(group):
        return {
            k: sum(q["grade"] == k for q in group) / len(group)
            for k in ("CORRECT", "WRONG", "ABSTAIN")
        }

    out["no_judge"] = {"answerable": base(ans), "unanswerable": base(imp)}

    def flagged_share(group: list[dict]) -> float | None:
        claims = [c for q in group for c in q["claims"]]
        return sum(c["supported"] < 0.5 for c in claims) / len(claims) if claims else None

    def any_flag(group: list[dict]) -> float | None:
        with_claims = [q for q in group if q["claims"]]
        return (
            sum(any(c["supported"] < 0.5 for c in q["claims"]) for q in with_claims)
            / len(with_claims)
            if with_claims
            else None
        )

    correct = [q for q in qs if q["grade"] == "CORRECT"]
    wrong = [q for q in qs if q["grade"] == "WRONG"]

    def weakest(group: list[dict]) -> list[float]:
        return [min(c["supported"] for c in q["claims"]) for q in group if q["claims"]]

    out["verify"] = {
        # Does the reply's least-supported claim separate wrong replies from correct ones?
        "auroc_correct_vs_wrong": auroc(weakest(correct), weakest(wrong)),
        "correct_answers": {
            "n": len(correct),
            "claims_flagged": flagged_share(correct),
            "answers_with_a_flag": any_flag(correct),
        },
        "wrong_answers": {
            "n": len(wrong),
            "claims_flagged": flagged_share(wrong),
            "answers_with_a_flag": any_flag(wrong),
        },
    }
    out["examples_wrong_unanswerable"] = [
        {"q": q["q"], "p_answerable": round(q["p_answerable"], 3), "response": q["response"][:300]}
        for q in imp
        if q["grade"] == "WRONG"
    ][:8]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(
        json.dumps({k: v for k, v in out.items() if k != "examples_wrong_unanswerable"}, indent=2)
    )


if __name__ == "__main__":
    main()
