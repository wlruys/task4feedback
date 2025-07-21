#!/bin/bash
#SBATCH --job-name=wandb-agent
#SBATCH --output=wandb-agent-%j.out
#SBATCH --error=wandb-agent-%j.err
#SBATCH --ntasks=1
#SBATCH --time=4:00:00

if [ -z "$WANDB_API_KEY" ]; then
  echo "Error: WANDB_API_KEY is not set."
  exit 1
fi

if [ $# -ne 3 ]; then
  echo "Usage: sbatch $0 <WANDB_SWEEP_ID> <WANDB_PROJECT_NAME> <WANDB_ENTITY>"
  exit 1
fi

SWEEP_ID=$1
PROJECT_NAME=$2
ENTITY_NAME=$3

echo "Launching wandb agent for sweep $SWEEP_ID on $HOSTNAME, project $PROJECT_NAME, entity $ENTITY_NAME"
wandb agent --project "$PROJECT_NAME" --entity "$ENTITY_NAME" "$SWEEP_ID"
