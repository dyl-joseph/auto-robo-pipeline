"""Run with python -m robo. All writes stay in the selected local store."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from robo.analysis import suggest
from robo.core import atomic_json
from robo.demo import make_demo
from robo.evaluation import evaluate
from robo.export import export_dataset, validate_export
from robo.ingest import import_episode
from robo.store import Store


def main():
    parser = argparse.ArgumentParser(description="Robot Episode Curation & Assisted Annotation")
    parser.add_argument("--store", type=Path, default=Path("data/local/demo"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="generate six explicitly synthetic episodes")
    imp = commands.add_parser("import", help="snapshot video and validated source manifest")
    imp.add_argument("manifest", type=Path)
    analysis = commands.add_parser("suggest", help="cache OpenCV heuristic motion candidates")
    analysis.add_argument("episode_id")
    curation = commands.add_parser("curate")
    curation.add_argument("--count", type=int, default=10)
    curation.add_argument("--policy", choices=["prioritized", "random"], default="prioritized")
    curation.add_argument("--seed", type=int, default=7)
    curation.add_argument("--output", type=Path, required=True)
    perception = commands.add_parser("perceive")
    perception.add_argument("episode_id")
    perception.add_argument("--model-dir", type=Path, default=Path("models/local/mobilenet-ssd"))
    perception.add_argument("--sample-every-s", type=float, default=0.5)
    perception.add_argument("--max-frames", type=int, default=600)
    ls_prepare = commands.add_parser("prepare-labelstudio")
    ls_prepare.add_argument("episode_id")
    ls_prepare.add_argument("--output", type=Path, default=Path("artifacts/labelstudio-bundle"))
    ls_prepare.add_argument("--fps", type=int, default=12)
    serve = commands.add_parser("serve", help="start the primary native Label Studio GUI")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--legacy-editor", action="store_true", help="development-only historical prototype"
    )
    export = commands.add_parser("export")
    export.add_argument("selection", type=Path)
    export.add_argument("output", type=Path)
    validate = commands.add_parser("validate-export")
    validate.add_argument("path", type=Path)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("protocol", type=Path)
    evaluation.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        store = Store(args.store)
        if args.command == "demo":
            result = [{"id": e["id"], "task": e["task"]} for e in make_demo(args.store)]
        elif args.command == "import":
            result = import_episode(args.manifest, store)
        elif args.command == "suggest":
            result = suggest(store, args.episode_id)
        elif args.command == "prepare-labelstudio":
            from robo.labelstudio import prepare_bundle

            result = prepare_bundle(store, args.episode_id, args.output, args.fps)
        elif args.command == "curate":
            from robo.curation import curate

            result = curate(store, args.count, args.policy, args.seed)
            atomic_json(args.output, result)
        elif args.command == "perceive":
            from robo.perception import perceive

            result = perceive(
                store, args.episode_id, args.model_dir, args.sample_every_s, args.max_frames
            )
        elif args.command == "export":
            result = export_dataset(store, json.loads(args.selection.read_text()), args.output)
        elif args.command == "validate-export":
            result = {"records": len(validate_export(args.path)), "valid": True}
        elif args.command == "evaluate":
            result = evaluate(json.loads(args.protocol.read_text()))
            atomic_json(args.output, result)
        else:
            if args.legacy_editor:
                import uvicorn

                from robo.api import create_app

                uvicorn.run(create_app(args.store), host="127.0.0.1", port=args.port)
            else:
                from robo.labelstudio import serve_labelstudio

                serve_labelstudio(store, args.port)
            return
        print(json.dumps(result, indent=2))
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
