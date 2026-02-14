# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cogfly is a personal learning project building a DQN (Deep Q-Network) agent to solve Gymnasium's LunarLander-v3, with plans to extend to multi-agent coordination using PettingZoo.

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run random baseline agent
python agents/random_agent.py

# Training (not yet implemented)
python train.py

# Evaluation (not yet implemented)
python evaluate.py
```

## Architecture

- **agents/**: Agent implementations (currently `random_agent.py` baseline)
- **utils/**: Shared utilities
- **train.py**: Training entrypoint (stub)
- **evaluate.py**: Evaluation entrypoint (stub)
- **notebooks/**: Jupyter notebooks for experimentation

## Key Details

- Uses **Gymnasium** (not legacy `gym`), with `LunarLander-v3`
- Modern Gymnasium API: `env.reset()` returns `(obs, info)`, `env.step()` returns `(obs, reward, terminated, truncated, info)`
- Episode termination: `done = terminated or truncated`
- PyTorch for neural networks, TensorBoard for training visualization
- Use Context7 MCP to check latest Gymnasium/PettingZoo docs when implementing new features
