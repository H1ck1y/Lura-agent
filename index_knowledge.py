"""Explicit admin entry: upsert only this project's separate knowledge collection."""

import json
import sys

from knowledge_tool import KnowledgeRAGTool
from phase1_config import InventoryError, Settings


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    tool = KnowledgeRAGTool(Settings.from_env())
    try:
        print(json.dumps(tool.index_documents(), ensure_ascii=False, indent=2))
    finally:
        tool.close()


if __name__ == "__main__":
    try:
        main()
    except InventoryError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
