# ws_aic

Intrinsic AI Challenge — SFP cable insertion with UR5e.

## View the policy

```bash
# Watch insertion in the viewer
.venv/bin/python3 scripts/eval_policy.py --arch mlp --weights data/outputs/mlp_policy_best.pt --viewer

# Headless eval with vision (97% success)
.venv/bin/python3 scripts/eval_policy.py --arch mlp --weights data/outputs/mlp_policy_best.pt --vision --vision-mode geometric --trials 20
```

## Submit to competition

```bash
cd src/aic/docker && docker compose build model

export AWS_PROFILE=aic
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com

docker tag my-solution:v1 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/balazs:v1
docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/balazs:v1
```

Then paste the URI on the submission portal.
