import wandb

api = wandb.Api()

run_id = "to8uyy5q"
try:
    run = api.run(f"danielstrauss-technical-university-of-munich/cleanRL/{run_id}")
    print("Still exists. State:", run.state)
    run.delete()
    print("Delete call sent.")
except wandb.errors.CommError as e:
    print("Confirmed deleted (or never existed):", e)
