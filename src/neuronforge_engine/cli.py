"""Command line interface for training, generating and serving Neuron Forge."""
import argparse
import sys

from .config import load_config
from .generation import generate
from .training import train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="neuronforge",
        description="Train and run a language model from random initialization",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train a new model from a text corpus")
    train_parser.add_argument("--config", default="configs/quick.yaml")
    train_parser.add_argument("--resume", action="store_true", help="resume from the configured checkpoint")

    gen_parser = commands.add_parser("generate", help="generate text from a trained checkpoint")
    gen_parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    gen_parser.add_argument("--prompt", default="")
    gen_parser.add_argument("--max-new-tokens", type=int, default=200)
    gen_parser.add_argument("--temperature", type=float, default=0.8)
    gen_parser.add_argument("--top-k", type=int, default=40)
    gen_parser.add_argument("--top-p", type=float, default=0.95)
    gen_parser.add_argument("--repetition-penalty", type=float, default=1.05)
    gen_parser.add_argument("--device", default="cpu")

    chat_parser = commands.add_parser("chat", help="start a local terminal chat session")
    chat_parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    chat_parser.add_argument("--max-new-tokens", type=int, default=160)
    chat_parser.add_argument("--temperature", type=float, default=0.8)
    chat_parser.add_argument("--device", default="cpu")

    serve_parser = commands.add_parser("serve", help="start the optional local HTTP and browser interface")
    serve_parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--device", default="cpu")

    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            train(load_config(args.config), resume=args.resume)
        elif args.command == "generate":
            print(generate(
                args.checkpoint, args.prompt, args.max_new_tokens, args.temperature,
                args.top_k, args.top_p, args.repetition_penalty, args.device,
            ))
        elif args.command == "chat":
            _terminal_chat(args)
        elif args.command == "serve":
            _serve(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _terminal_chat(args) -> None:
    print("Neuron Forge local chat. Type /exit to quit.")
    history = ""
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in {"/exit", "/quit"}:
            break
        if not message:
            continue
        prompt = (history + f"\nUser: {message}\nAssistant:")[-4000:]
        answer = generate(args.checkpoint, prompt, args.max_new_tokens, args.temperature)
        answer = answer[len(prompt):].strip() if answer.startswith(prompt) else answer.strip()
        print(f"Neuron Forge: {answer}\n")
        history = (history + f"\nUser: {message}\nAssistant: {answer}")[-4000:]


def _serve(args) -> None:
    try:
        import uvicorn
        from .api import create_app
    except ImportError as exc:
        raise RuntimeError("Install web dependencies with: pip install -e '.[web]'") from exc
    app = create_app(args.checkpoint, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
