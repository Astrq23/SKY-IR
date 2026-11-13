"""python -m skyir [--host 0.0.0.0] [--port 8000] [--model NAME] [--no-warmup]"""
import argparse
import logging

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(prog="skyir")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=None, help="model id to load at startup (default: first in models/)")
    ap.add_argument("--no-warmup", action="store_true", help="load the model lazily on first use")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from .server import app, get_detector

    if not a.no_warmup:
        get_detector(a.model)  # builds the TensorRT engine on first run
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
