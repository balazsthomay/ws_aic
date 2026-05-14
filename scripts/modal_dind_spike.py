"""Spike test: does Modal Sandbox alpha DinD work for docker compose?

Run: uv run modal run scripts/modal_dind_spike.py

Cost: ~$0.05 (8-CPU box for ~5 min). Tells us if the v38 collection plan
can use Modal Sandbox + DinD or needs to fall back to combined-image.
"""

import modal

# Per docs (https://modal.com/docs/guide/docker-in-sandboxes), DinD requires
# image builder version 2025.06+. Modal client picks this up from the project
# config; we set it on the image as a sanity check.
DOCKER_IMAGE = (
    modal.Image.debian_slim()
    .apt_install(
        "docker.io",
        "docker-compose",   # legacy compose CLI
        "ca-certificates", "curl", "gnupg", "less", "iproute2",
    )
    .pip_install("requests")
)

app = modal.App("aic-dind-spike", image=DOCKER_IMAGE)


@app.local_entrypoint()
def main():
    """Create a sandbox with experimental DinD enabled, probe Docker."""
    print("Creating sandbox with enable_docker=True ...")
    sb = modal.Sandbox.create(
        image=DOCKER_IMAGE,
        cpu=2.0,
        memory=4096,
        timeout=600,
        experimental_options={"enable_docker": True},
        app=app,
    )
    try:
        print(f"Sandbox started: {sb.object_id}")

        def run(cmd: list[str], label: str) -> int:
            print(f"\n=== {label} ===")
            print(f"$ {' '.join(cmd)}")
            p = sb.exec(*cmd)
            stdout = p.stdout.read()
            stderr = p.stderr.read()
            rc = p.wait()
            if stdout:
                print(stdout)
            if stderr:
                print(f"[stderr] {stderr}")
            print(f"rc={rc}")
            return rc

        # 1. What's pre-configured by enable_docker=True?
        run(["docker", "--version"], "docker --version")
        run(["bash", "-c", "ls -la /var/run/docker.sock 2>&1 || echo no-sock"],
            "/var/run/docker.sock")
        run(["bash", "-c", "ls /etc/init.d/ 2>&1 | head -20"], "/etc/init.d listing")
        run(["bash", "-c", "which dockerd; which docker-init"], "binary locations")
        run(["bash", "-c", "ps auxf | head -30"], "running processes")
        run(["bash", "-c",
             "find / -name '*dockerd*' -o -name 'modal-docker*' 2>/dev/null | head -20"],
            "search for docker init scripts")

        # 2. Try dockerd with iptables disabled (gVisor blocks nft).
        run(["bash", "-c",
             "nohup dockerd --iptables=false --bridge=none --ip-forward=false "
             "  > /tmp/dockerd.log 2>&1 & echo $!; sleep 8; "
             "echo '--- TAIL DOCKERD LOG ---'; tail -30 /tmp/dockerd.log; "
             "echo '--- INFO ---'; docker info 2>&1 | head -40 || true"],
            "dockerd --iptables=false --bridge=none start + check")

        # 4. Try a trivial container with --network=host (only mode that works
        # with --bridge=none).
        rc = run(
            ["docker", "run", "--rm", "--network=host",
             "alpine:latest", "echo", "hello-from-nested"],
            "docker run --network=host alpine echo",
        )
        if rc == 0:
            print("\n>>> DinD WORKS. Proceed with full Modal harness. <<<")
        else:
            print("\n>>> DinD does NOT work for `docker run`. Fall back to combined-image. <<<")

        # 5. If we got this far, sanity-check compose with a 2-service stack
        # using host networking (the only mode that works with --bridge=none).
        if rc == 0:
            sb.exec(
                "bash", "-c",
                "mkdir -p /work && cd /work && cat > docker-compose.yaml <<'EOF'\n"
                "version: '3'\n"
                "services:\n"
                "  a:\n"
                "    image: alpine:latest\n"
                "    network_mode: host\n"
                "    command: ['sh', '-c', 'echo a-side; sleep 2']\n"
                "  b:\n"
                "    image: alpine:latest\n"
                "    network_mode: host\n"
                "    command: ['sh', '-c', 'echo b-side; sleep 2']\n"
                "EOF\n"
            ).wait()
            run(
                ["docker-compose", "-f", "/work/docker-compose.yaml",
                 "up", "--abort-on-container-exit"],
                "docker-compose up (2 alpine services, network_mode: host)",
            )

        print("\n=== Spike complete ===")
    finally:
        sb.terminate()
