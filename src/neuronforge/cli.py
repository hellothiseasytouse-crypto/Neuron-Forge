import argparse
from .config import load_config
from .train import train
from .generate import generate


def main() -> None:
    p = argparse.ArgumentParser(prog="neuronforge")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--config", default="configs/tiny.yaml")

    g = sub.add_parser("generate")
    g.add_argument("--checkpoint", default="checkpoints/latest.pt")
    g.add_argument("--prompt", default="")
    g.add_argument("--max-new-tokens", type=int, default=200)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-k", type=int, default=None)

    a = p.parse_args()
    if a.cmd == "train":
        train(load_config(a.config))
    else:
        print(generate(a.checkpoint, a.prompt, a.max_new_tokens, a.temperature, a.top_k))


if __name__ == "__main__":
    main()
