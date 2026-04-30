import argparse
import json
from pathlib import Path
from typing import Any


def build_block(context: str, dockerfile: str) -> str:
    return f"""build:
      context: {context}
      dockerfile: {dockerfile}"""


def build_header(build_context: str, dockerfile: str, image_name: str) -> str:
    return f"""name: fl-backend

services:
  superlink:
    image: flwr/superlink:1.27.0
    command:
      - --insecure
      - --serverappio-api-address
      - 0.0.0.0:9091
      - --fleet-api-address
      - 0.0.0.0:9092
      - --fleet-api-type
      - grpc-rere
    ports:
      - "9093:9093"
      - "9091:9091"
      - "9092:9092"
    cpus: "0.25"
    mem_limit: 512m


  superexec-serverapp:
    image: {image_name}
    build:
      context: {build_context}
      dockerfile: {dockerfile}
    command:
      - --insecure
      - --plugin-type
      - serverapp
      - --appio-api-address
      - superlink:9091
    depends_on:
      - superlink
    stop_signal: SIGINT
    cpus: "1.0"
    mem_limit: 2g

    volumes:
      - {build_context}/checkpoints:/app/checkpoints
      - {build_context}/plotting/saved_plots:/app/plotting/saved_plots

"""


def generate_supernode(node_num: int, partition_id: int, port: int, num_clients: int) -> str:
    return f"""  supernode-{node_num}:
    image: flwr/supernode:1.27.0
    command:
      - --insecure
      - --superlink
      - superlink:9092
      - --clientappio-api-address
      - 0.0.0.0:{port}
      - --isolation
      - process
      - --node-config
      - "partition-id={partition_id} num-partitions={num_clients}"
    depends_on:
      - superlink
    cpus: "0.25"
    mem_limit: 512m
  

"""


def generate_resource_block(resources: dict[str, Any]) -> str:
    lines: list[str] = []

    cpus = resources.get("cpus")
    cpuset = resources.get("cpuset")
    mem_limit = resources.get("mem_limit")
    mem_reservation = resources.get("mem_reservation")
    pids_limit = resources.get("pids_limit")
    shm_size = resources.get("shm_size")
    ulimits = resources.get("ulimits")
    group_add = resources.get("group_add")
    volumes = resources.get("volumes")

    if cpus is not None:
        lines.append(f'cpus: "{cpus}"')

    if cpuset is not None:
        lines.append(f'cpuset: "{cpuset}"')

    if mem_limit is not None:
        lines.append(f"mem_limit: {mem_limit}")

    if mem_reservation is not None:
        lines.append(f"mem_reservation: {mem_reservation}")

    if pids_limit is not None:
        lines.append(f"pids_limit: {pids_limit}")

    if shm_size is not None:
        lines.append(f"shm_size: {shm_size}")

    if group_add:
        lines.append("group_add:")
        for group in group_add:
            lines.append(f"  - {group}")

    if volumes:
        lines.append("volumes:")
        for volume in volumes:
            lines.append(f"  - {volume}")

    if ulimits:
        lines.append("ulimits:")
        for key, value in ulimits.items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                if "soft" in value:
                    lines.append(f"    soft: {value['soft']}")
                if "hard" in value:
                    lines.append(f"    hard: {value['hard']}")
            else:
                lines.append(f"  {key}: {value}")

    if not lines:
        return ""

    return "\n".join(f"    {line}" for line in lines) + "\n"


def generate_clientapp(
    node_num: int,
    port: int,
    image_name: str,
    resources: dict[str, Any],
) -> str:
    resource_block = generate_resource_block(resources)

    return f"""  superexec-clientapp-{node_num}:
    image: {image_name}
    command:
      - --insecure
      - --plugin-type
      - clientapp
      - --appio-api-address
      - supernode-{node_num}:{port}
    depends_on:
      - supernode-{node_num}
    stop_signal: SIGINT
{resource_block}
"""


def load_overrides(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merge_resources(defaults: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)

    for key, value in override.items():
        if key == "ulimits" and isinstance(value, dict):
            base_ulimits = dict(merged.get("ulimits", {}))
            base_ulimits.update(value)
            merged["ulimits"] = base_ulimits
        else:
            merged[key] = value

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Docker Compose file for Flower clients.")

    parser.add_argument("--num-clients", type=int, default=1)
    parser.add_argument("--start-port", type=int, default=9094)
    parser.add_argument("--output", type=str, default="compose.yml")

    parser.add_argument("--build-context", type=str, default="./fl-backend")
    parser.add_argument("--dockerfile", type=str, default="Dockerfile")

    parser.add_argument("--cpus", type=str, default="2.0")
    parser.add_argument("--cpuset", type=str, default=None)
    parser.add_argument("--mem-limit", type=str, default="8g")
    parser.add_argument("--mem-reservation", type=str, default=None)
    parser.add_argument("--pids-limit", type=int, default=None)
    parser.add_argument("--shm-size", type=str, default=None)

    parser.add_argument(
        "--client-overrides",
        type=str,
        default=None,
        help="Path to JSON file with per-client overrides",
    )

    parser.add_argument(
        "--image-name",
        type=str,
        default="fl-backend-app:latest",
        help="Shared image name for server/client app",
    )

    args = parser.parse_args()

    if args.num_clients < 1:
        raise ValueError("num-clients must be at least 1")

    default_resources = {
        "cpus": args.cpus,
        "cpuset": args.cpuset,
        "mem_limit": args.mem_limit,
        "mem_reservation": args.mem_reservation,
        "pids_limit": args.pids_limit,
        "shm_size": args.shm_size,
        "group_add": ["render", "video"],
        "volumes": [
            f"{args.build_context}/datasets:/app/datasets",
        ],
    }

    overrides = load_overrides(args.client_overrides)

    parts = [build_header(args.build_context, args.dockerfile, args.image_name)]

    for i in range(args.num_clients):
        node_num = i + 1
        partition_id = i
        port = args.start_port + i

        client_override = overrides.get(str(node_num), {})
        resources = merge_resources(default_resources, client_override)

        parts.append(generate_supernode(node_num, partition_id, port, args.num_clients))
        parts.append(generate_clientapp(node_num, port, args.image_name, resources)) 

    Path(args.output).write_text("".join(parts), encoding="utf-8")

    print(
        f"Generated {args.output} with {args.num_clients} clients "
        f"using build context {args.build_context}."
    )


if __name__ == "__main__":
    main()