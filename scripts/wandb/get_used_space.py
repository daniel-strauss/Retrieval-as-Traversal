import wandb

api = wandb.Api()
runs = api.runs("danielstrauss-technical-university-of-munich/cleanRL")

total = 0
for i, run in enumerate(runs):
    for f in run.files():
        total += f.size or 0
    for art in run.logged_artifacts():
        total += art.size or 0
    print(f"Inspecting run {i} of {len(runs)}. Current size: {total / 1024**3:.2f} GiB.")
print(f"{total / 1024**3:.2f} GiB across {len(runs)} runs")