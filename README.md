# HumbleMeister

A transformer chess engine that treats chess as a language modelling problem. A game is a sequence of UCI move tokens; the model is trained to predict the next move — no board evaluation function, no tree search, no hand-crafted features.

The model has a policy head (move prediction) and a value head (board evaluation), trained jointly on human games and self-play with Stockfish-derived supervision.

A Gradio web app (`app.py`) lets you play against the engine or watch it play itself. A live instance is running at [huggingface.co/spaces/StanStanStan3141592/HumbleMeister](https://huggingface.co/spaces/StanStanStan3141592/HumbleMeister).

---

## Documentation

- [MODEL.md](MODEL.md) — architecture, tokenization, training loop, data pipeline, inference, and hyperparameters
- [MOVE_PICKING.md](MOVE_PICKING.md) — inference-time move selection: mate-in-1 short-circuit, value-gap masking, and the self-play temperature anneal
- [SELF_PLAY_LOSS.md](SELF_PLAY_LOSS.md) — loss function design for self-play games and the policy/value weighting modes
- [FLASH_ATTENTION.md](FLASH_ATTENTION.md) — Flash Attention integration and the length-bucketed batch sampler that eliminates padding contamination
