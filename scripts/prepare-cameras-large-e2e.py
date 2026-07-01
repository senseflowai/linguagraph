#!/usr/bin/env python3
"""Build the cameras e2e graph from a raw cameras dump.

The source dump is intentionally not checked into the repository. This script
turns it into the compact GraphBuilder JSON shape used by `linguagraph-e2e`.
The generated output belongs under `target/e2e/`, which is already ignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODULE_CODES = {
    "fr": "FaceRecognitionModule",
    "tp": "TamperingModule",
    "sa": "SituationAnalyticsModule",
    "lpr": "LicensePlateRecognitionModule",
    "ao": "AbandonedObjectsModule",
}


def typed(kind: str, value: Any) -> dict[str, Any]:
    return {"type": kind, "value": value}


def keyword(value: Any) -> dict[str, Any]:
    return typed("Keyword", "" if value is None else str(value))


def number(value: Any) -> dict[str, Any]:
    return typed("Number", value)


def boolean(value: Any) -> dict[str, Any]:
    return typed("Boolean", bool(value))


def clean_id(prefix: str, value: Any) -> str:
    raw = "" if value is None else str(value)
    out = []
    for ch in raw:
        if ch.isascii() and (ch.isalnum() or ch in "-_"):
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    return f"{prefix}-{cleaned or 'unknown'}"


def canonical(entity_type: str, props: dict[str, dict[str, Any]]) -> str:
    parts = [f"type: {entity_type}"]
    for key in sorted(k for k in props if k != "_canonical"):
        value = props[key]["value"]
        if value is None or value == "":
            continue
        parts.append(f"{key}: {value}")
    return "\n".join(parts)


def entity(local_id: str, entity_type: str, props: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": local_id,
        "type": entity_type,
        "labels": ["camera_large_domain"],
        "primary_key": "id",
        "properties": props,
    }


def camera_modules(item: dict[str, Any]) -> list[str]:
    names = []
    for module in ((item.get("video_analytics") or {}).get("modules") or []):
        name = module.get("module")
        if name:
            names.append(str(name))
    for code in item.get("video_modules") or []:
        mapped = MODULE_CODES.get(str(code), str(code))
        if mapped:
            names.append(mapped)
    return sorted(set(names))


def build_graph(items: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, int]]:
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    places: dict[str, dict[str, Any]] = {}
    modules: dict[str, dict[str, Any]] = {}

    stats = {
        "cameras": 0,
        "places": 0,
        "modules": 0,
        "located_at": 0,
        "uses_module": 0,
        "active": 0,
        "inactive": 0,
    }

    for item in items:
        camera_id = item.get("id")
        if not camera_id:
            continue

        origin = item.get("origin") or {}
        place_raw_id = origin.get("place_id") or item.get("place_id") or origin.get("place_name")
        place_local_id = clean_id("place", place_raw_id)
        place_name = origin.get("place_name") or str(place_raw_id)

        if place_local_id not in places:
            place_props = {
                "id": keyword(place_local_id),
                "external_id": keyword(place_raw_id),
                "name": keyword(place_name),
                "description": keyword(origin.get("place_description")),
            }
            if origin.get("place_latitude") is not None:
                place_props["latitude"] = number(origin.get("place_latitude"))
            if origin.get("place_longitude") is not None:
                place_props["longitude"] = number(origin.get("place_longitude"))
            places[place_local_id] = entity(place_local_id, "Place", place_props)

        cam_local_id = clean_id("camera", camera_id)
        cam_props = {
            "id": keyword(camera_id),
            "name": keyword(item.get("name")),
            "state": keyword(item.get("state")),
            "state_reason": keyword(item.get("state_reason")),
            "is_dvr": boolean(item.get("is_dvr")),
            "edge": boolean(item.get("edge")),
            "place_id": keyword(place_raw_id),
            "created_at": keyword(item.get("created_at")),
            "updated_at": keyword(item.get("updated_at")),
            "scheduler_group": keyword(item.get("scheduler_group")),
            "capacitygroup_token": keyword(item.get("capacitygroup_token")),
        }
        if (item.get("video_streaming") or {}).get("storage_depth") is not None:
            cam_props["storage_depth"] = number(item["video_streaming"]["storage_depth"])
        host = ((item.get("source") or {}).get("body") or {}).get("host")
        if host:
            cam_props["source_host"] = keyword(host)

        entities.append(entity(cam_local_id, "Camera", cam_props))
        relations.append({"from": cam_local_id, "to": place_local_id, "type": "LOCATED_AT"})

        stats["cameras"] += 1
        if item.get("state") == "active":
            stats["active"] += 1
        if item.get("state") == "inactive":
            stats["inactive"] += 1

        for module_name in camera_modules(item):
            module_local_id = clean_id("module", module_name)
            if module_local_id not in modules:
                modules[module_local_id] = entity(
                    module_local_id,
                    "AnalyticsModule",
                    {
                        "id": keyword(module_local_id),
                        "name": keyword(module_name),
                    },
                )
            relations.append({"from": cam_local_id, "to": module_local_id, "type": "USES_MODULE"})
            stats["uses_module"] += 1

    entities.extend(places.values())
    entities.extend(modules.values())
    stats["places"] = len(places)
    stats["modules"] = len(modules)
    stats["located_at"] = stats["cameras"]

    graph = {"entities": entities, "relations": relations}
    return graph, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/df/Downloads/cameras_dump.json")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", default="target/e2e/cameras_1k.graph.json")
    parser.add_argument("--stats", default="target/e2e/cameras_1k.stats.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_path = Path(args.stats)

    with input_path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise SystemExit(f"{input_path} must contain a JSON array")
    items = items[: max(0, args.limit)]

    graph, stats = build_graph(items)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    print(
        "generated "
        f"{output_path} cameras={stats['cameras']} places={stats['places']} "
        f"modules={stats['modules']} relations={len(graph['relations'])}"
    )


if __name__ == "__main__":
    main()
