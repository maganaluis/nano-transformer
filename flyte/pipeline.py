from flytekit import task, workflow, Resources
from flytekit.configuration import Image

@task(
    container_image=Image(
        python_image="your-docker-registry/dummy_transformer:latest"  # Update with your image name
    ),
    requests=Resources(cpu="1", mem="2Gi"),
)
def train_task(max_epochs: int, batch_size: int) -> str:
    import os
    # Run the job with Hydra argument overrides
    os.system(f"python main.py trainer.max_epochs={max_epochs} trainer.batch_size={batch_size}")
    return "Training Completed"

@workflow
def dummy_transformer_workflow(max_epochs: int = 10, batch_size: int = 32) -> str:
    return train_task(max_epochs=max_epochs, batch_size=batch_size)

if __name__ == "__main__":
    print(dummy_transformer_workflow())
