"""Run with python -m robo. All writes stay in the selected local store."""

import argparse
import json
import os
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
    ls_prepare.add_argument(
        "--fps", type=int, help="explicit coarser grid; default retains source frames"
    )
    native = commands.add_parser(
        "native-review", help="fresh native imports and explicit decisions"
    )
    native.add_argument("--instance", default="http://127.0.0.1:8000")
    native_commands = native.add_subparsers(dest="native_action", required=True)
    bind = native_commands.add_parser("bind", help="register a prepared task once")
    bind.add_argument("bundle", type=Path, help="prepared bundle.json")
    bind.add_argument("--project", type=int, required=True)
    bind.add_argument("--task", type=int, required=True)
    for name in ("refresh", "accept", "reject"):
        action = native_commands.add_parser(name)
        action.add_argument("episode_id")
        action.add_argument("--expected-revision", type=int, required=True)
        action.add_argument("--actor", required=True)
        action.add_argument("--reason", required=True)
        if name == "refresh":
            action.add_argument("--annotation", type=int, required=True)
    inspect = native_commands.add_parser("inspect", help="read immutable local review history")
    inspect.add_argument("episode_id")
    native_export = native_commands.add_parser("export", help="check native freshness and export")
    native_export.add_argument("selection", type=Path)
    native_export.add_argument("output", type=Path)
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
        elif args.command == "native-review":
            if args.native_action == "inspect":
                result = {
                    "latest": store.latest(args.episode_id),
                    "history": store.history(args.episode_id),
                }
            else:
                from label_studio_sdk import LabelStudio

                from robo.native_review import NativeReviewBridge, local_instance

                instance = local_instance(args.instance)
                key = os.environ.get("LABEL_STUDIO_API_KEY")
                if not key:
                    raise ValueError(
                        "set LABEL_STUDIO_API_KEY to an existing authorized credential, "
                        "or use NativeReviewBridge with an authorized SDK session; none is created"
                    )
                sdk = LabelStudio(base_url=instance, api_key=key, timeout=10)
                bridge = NativeReviewBridge(sdk, store, instance)
                if args.native_action == "bind":
                    result = bridge.register(
                        json.loads(args.bundle.read_text()), args.project, args.task
                    )
                elif args.native_action == "refresh":
                    result = bridge.refresh(
                        args.episode_id,
                        args.annotation,
                        args.expected_revision,
                        args.actor,
                        args.reason,
                    )
                elif args.native_action == "export":
                    result = bridge.export(json.loads(args.selection.read_text()), args.output)
                else:
                    result = bridge.decide(
                        args.episode_id,
                        args.expected_revision,
                        args.native_action,
                        args.actor,
                        args.reason,
                    )
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
    except Exception as exc:
        # SDK exception bodies can include response headers; never print credentials.
        if args.command == "native-review" and hasattr(exc, "status_code"):
            print(
                f"Native API request failed (HTTP {exc.status_code}); no decision recorded",
                file=sys.stderr,
            )
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
