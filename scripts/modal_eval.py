"""Run the AIC eval stack on Modal (real linux/amd64) and return scores.

Usage:
    uv run modal run scripts/modal_eval.py --model-tag v8

Architecture:
    A single Modal Function with docker-in-docker. Pulls aic_eval +
    my-solution:<tag> from ECR, runs the same compose stack we use
    locally, scrapes scores from the engine output.
"""

import os

import modal

app = modal.App("aic-eval")

# DinD-capable image. Docker daemon will be started inside the container.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("docker.io", "awscli", "curl", "ca-certificates", "iptables", "uidmap")
)


# ECR creds via `modal secret create aic-ecr ...` (one-time setup).
ecr_secret = modal.Secret.from_name("aic-ecr")


@app.function(
    image=image,
    timeout=1800,
    cpu=4.0,
    memory=8192,
    gpu="T4",  # GPU functions get more kernel capabilities
    secrets=[ecr_secret],
)
def run_eval(model_tag: str = "v8") -> str:
    """Run the AIC eval against my-solution:<model_tag>. Returns full log."""
    import subprocess, time, sys

    # 1) Try dockerd with various tweaks for unprivileged-ish operation
    print("[modal_eval] env:", flush=True)
    subprocess.run(["id"], check=False)
    subprocess.run(["bash", "-c", "cat /proc/self/status | grep -E 'Cap|Uid|Gid' | head -10"], check=False)
    subprocess.run(["bash", "-c", "ls /dev/fuse 2>&1"], check=False)

    print("[modal_eval] starting dockerd (vfs storage, no userland-proxy)...", flush=True)
    dockerd = subprocess.Popen(
        ["dockerd",
         "--storage-driver=vfs",
         "--iptables=false",
         "--bridge=none",
         "--userland-proxy=false",
         "--host=unix:///var/run/docker.sock"],
        stdout=open("/tmp/dockerd.log", "w"),
        stderr=subprocess.STDOUT,
    )
    for i in range(30):
        if subprocess.run(["docker", "info"], capture_output=True).returncode == 0:
            print(f"[modal_eval] dockerd ready after {i}s", flush=True)
            break
        time.sleep(1)
    else:
        # Show why dockerd failed
        with open("/tmp/dockerd.log") as f:
            print("=== dockerd log ===")
            print(f.read()[-3000:])
        dockerd.terminate()
        raise RuntimeError("dockerd failed to start within 30s")

    # 2) ECR login
    print("[modal_eval] authenticating to ECR...", flush=True)
    pwd = subprocess.run(
        ["aws", "ecr", "get-login-password", "--region", "us-east-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin",
         "973918476471.dkr.ecr.us-east-1.amazonaws.com"],
        input=pwd, text=True, check=True,
    )

    # 3) Pull both images in parallel
    print("[modal_eval] pulling images...", flush=True)
    eval_pull = subprocess.Popen(["docker", "pull", "ghcr.io/intrinsic-dev/aic/aic_eval:latest"])
    model_pull = subprocess.Popen([
        "docker", "pull",
        f"973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/balazs:{model_tag}",
    ])
    eval_pull.wait()
    model_pull.wait()
    if eval_pull.returncode != 0 or model_pull.returncode != 0:
        raise RuntimeError(f"image pull failed: eval={eval_pull.returncode}, model={model_pull.returncode}")

    # 4) Create internal network + start eval (background) + start model (foreground)
    eval_image = "ghcr.io/intrinsic-dev/aic/aic_eval"
    model_image = f"973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/balazs:{model_tag}"

    subprocess.run(["docker", "network", "rm", "aic_default"], capture_output=True)
    subprocess.run(["docker", "network", "create", "--internal", "aic_default"], check=True)

    print("[modal_eval] starting eval container...", flush=True)
    subprocess.run([
        "docker", "run", "-d",
        "--name", "eval",
        "--network", "aic_default",
        "--network-alias", "eval",
        "-e", "AIC_EVAL_PASSWD=CHANGE_IN_PROD",
        "-e", "AIC_MODEL_PASSWD=CHANGE_IN_PROD",
        "-e", "AIC_ENABLE_ACL=true",
        eval_image,
        "gazebo_gui:=false", "launch_rviz:=false", "ground_truth:=false",
        "start_aic_engine:=true", "shutdown_on_aic_engine_exit:=true",
    ], check=True)

    print("[modal_eval] starting model container (foreground)...", flush=True)
    model_proc = subprocess.Popen([
        "docker", "run",
        "--name", "model",
        "--network", "aic_default",
        "-e", "RMW_IMPLEMENTATION=rmw_zenoh_cpp",
        "-e", "ZENOH_ROUTER_CHECK_ATTEMPTS=-1",
        "-e", "AIC_ROUTER_ADDR=eval:7447",
        "-e", "AIC_MODEL_PASSWD=CHANGE_IN_PROD",
        "-e", "AIC_ENABLE_ACL=true",
        model_image,
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    eval_log_proc = subprocess.Popen(
        ["docker", "logs", "-f", "eval"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    log_lines: list[str] = []

    import threading
    def reader(p, prefix):
        for line in p.stdout:
            tagged = f"{prefix} | {line}"
            sys.stdout.write(tagged)
            sys.stdout.flush()
            log_lines.append(tagged)

    threads = [
        threading.Thread(target=reader, args=(model_proc, "model"), daemon=True),
        threading.Thread(target=reader, args=(eval_log_proc, "eval"), daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        eval_wait = subprocess.run(
            ["docker", "wait", "eval"], capture_output=True, text=True, timeout=1500,
        )
        print(f"[modal_eval] eval exited with code {eval_wait.stdout.strip()}", flush=True)
    finally:
        subprocess.run(["docker", "rm", "-f", "eval", "model"], capture_output=True)
        subprocess.run(["docker", "network", "rm", "aic_default"], capture_output=True)
        for t in threads:
            t.join(timeout=5)

    return "".join(log_lines)


@app.local_entrypoint()
def main(model_tag: str = "v8"):
    """uv run modal run scripts/modal_eval.py --model-tag v8"""
    log = run_eval.remote(model_tag)
    print("\n\n========== SCORE SUMMARY ==========")
    for line in log.splitlines():
        if any(k in line for k in ("Score:", "tier_", "Trial '", "score:", "FATAL", "ERROR")):
            print(line)
