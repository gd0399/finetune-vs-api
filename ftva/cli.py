"""Command line.

  python -m ftva.cli generate
  python -m ftva.cli eval --spec anthropic:claude-haiku-5-5 --label haiku
  python -m ftva.cli eval --spec openai:Qwen/Qwen3-8B@http://localhost:8000/v1 --label qwen3-8b-base --workers 64
  python -m ftva.cli eval --spec openai:ft@http://localhost:8000/v1 --label qwen3-8b-lora --workers 64
  python -m ftva.cli report --gpu-hourly 0.80
"""
import argparse

from . import backends, generate, report, run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ftva")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="build the synthetic dataset")
    g.add_argument("--n-train", type=int, default=2000)
    g.add_argument("--n-test", type=int, default=300)
    g.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval", help="run one model on the test splits")
    e.add_argument("--spec", required=True, help="anthropic:<model> | openai:<model>@<url> | mock[:err]")
    e.add_argument("--label", required=True, help="name shown in the report")
    e.add_argument("--splits", default="test_id,test_ood")
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--limit", type=int)
    r = sub.add_parser("report", help="score all runs, write results/REPORT.md and charts")
    r.add_argument("--gpu-hourly", type=float, default=0.80, help="your GPU rental price, USD/hour")
    r.add_argument("--train-hours", type=float, default=1.5)
    r.add_argument("--batch", action="store_true", help="price the API at Batch API rates (50%% off)")
    r.add_argument("--price", action="append", default=[], help="override: label=api:<model> or label=gpu")
    a = ap.parse_args(argv)

    if a.cmd == "generate":
        print(generate.generate(n_train=a.n_train, n_test=a.n_test, seed=a.seed))
    elif a.cmd == "eval":
        b = backends.make(a.spec)
        for s in a.splits.split(","):
            run.run(b, a.label, s, workers=a.workers, limit=a.limit, spec=a.spec)
    elif a.cmd == "report":
        pricing = dict(p.split("=", 1) for p in a.price)
        print(report.write(gpu_hourly=a.gpu_hourly, train_hours=a.train_hours, batch=a.batch, pricing=pricing))


if __name__ == "__main__":
    main()
