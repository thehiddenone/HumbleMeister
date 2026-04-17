from __future__ import annotations

import chess
import chess.svg
import gradio as gr
from huggingface_hub import snapshot_download

from humblemeister import ChessGame, ChessModel

# Add HF repos here
MODEL_REGISTRY: dict[str, str] = {
    "688M params, no self-play": "StanStanStan3141592/hmeister_giant_04072026_2000_b3",
}

# One ChessModel per model name — weights are shared across all sessions
_models: dict[str, ChessModel] = {}

def _btn_start() -> dict:
    return gr.update(value="Start game", variant="primary")


def _btn_surrender() -> dict:
    return gr.update(value="Surrender", variant="stop")


def _resolve_path(model_name: str) -> str:
    """Download model from HF Hub if needed and return local cache path."""
    repo_id = MODEL_REGISTRY[model_name]
    return snapshot_download(repo_id=repo_id)


def _get_model(model_name: str) -> ChessModel:
    if model_name not in _models:
        path = _resolve_path(model_name)
        _models[model_name] = ChessModel.load(path, device="cpu")
    return _models[model_name]


def _board_svg(board: chess.Board, player_color: chess.Color) -> str:
    lastmove = board.peek() if board.move_stack else None
    flipped = player_color == chess.BLACK
    return chess.svg.board(board, lastmove=lastmove, size=400, flipped=flipped)  # type: ignore


def start_or_surrender(
    model_name: str,
    color_choice: str,
    game: ChessGame | None,
) -> tuple[str, str, str, ChessGame | None, dict]:
    """Start a new game, or surrender the current one if a game is in progress."""
    player_color = chess.WHITE if color_choice == "White" else chess.BLACK

    # Surrender path: game is active and not yet over
    if game is not None and not game.board.is_game_over():
        board_html = _board_svg(game.board, player_color)
        return board_html, color_choice, "You surrendered. You lost.", None, _btn_start()

    # Start game path
    if not model_name:
        board_html = _board_svg(game.board, player_color) if game else ""
        return board_html, color_choice, "Please select a model first.", game, _btn_start()

    chess_model = _get_model(model_name)
    game = ChessGame(chess_model, value_weight=1.0, use_kv_cache=True)
    game.start_game(player_color)

    status = "Your turn." if player_color == chess.WHITE else "Model played. Your turn."
    return _board_svg(game.board, player_color), color_choice, status, game, _btn_surrender()


def play_move(
    color_choice: str,
    move_input: str,
    game: ChessGame | None,
) -> tuple[str, str, str, ChessGame | None, dict]:
    """Called when the player submits a move."""
    if game is None:
        return "", color_choice, "Please select a model and start a game first.", None, _btn_start()

    player_color = chess.WHITE if color_choice == "White" else chess.BLACK

    if game.board.is_game_over():
        return _board_svg(game.board, player_color), color_choice, f"Game over — {game.board.result()}. Start a new game.", game, _btn_start()

    # parse UCI or SAN
    text = move_input.strip()
    move = None
    try:
        candidate = chess.Move.from_uci(text)
        if candidate in game.board.legal_moves:
            move = candidate
    except ValueError:
        pass
    if move is None:
        try:
            move = game.board.parse_san(text)
        except ValueError:
            return _board_svg(game.board, player_color), color_choice, f"Invalid move: '{text}'. Use UCI (e.g. e2e4) or SAN (e.g. e4, Nf3, O-O).", game, gr.update()

    # apply player move
    game.apply_move(move)

    if game.board.is_game_over():
        return _board_svg(game.board, player_color), color_choice, f"Game over — {game.board.result()}", game, _btn_start()

    # model responds
    model_move = game.sample_move()
    model_san = game.board.san(model_move)
    game.apply_move(model_move)

    if game.board.is_game_over():
        status = f"Model played {model_san}. Game over — {game.board.result()}"
        return _board_svg(game.board, player_color), color_choice, status, game, _btn_start()

    status = f"Model played {model_san}. Your turn."
    return _board_svg(game.board, player_color), color_choice, status, game, gr.update()


def play_for_me(
    color_choice: str,
    game: ChessGame | None,
) -> tuple[str, str, str, ChessGame | None, dict]:
    """Model plays the player's move, then responds as the opponent."""
    if game is None:
        return "", color_choice, "Please select a model and start a game first.", None, _btn_start()

    player_color = chess.WHITE if color_choice == "White" else chess.BLACK

    if game.board.is_game_over():
        return _board_svg(game.board, player_color), color_choice, f"Game over — {game.board.result()}. Start a new game.", game, _btn_start()

    # model plays as the player
    player_move = game.sample_move()
    player_san = game.board.san(player_move)
    game.apply_move(player_move)

    if game.board.is_game_over():
        status = f"Played {player_san} for you. Game over — {game.board.result()}"
        return _board_svg(game.board, player_color), color_choice, status, game, _btn_start()

    # model responds as the opponent
    opponent_move = game.sample_move()
    opponent_san = game.board.san(opponent_move)
    game.apply_move(opponent_move)

    if game.board.is_game_over():
        status = f"Played {player_san} for you. Opponent played {opponent_san}. Game over — {game.board.result()}"
        return _board_svg(game.board, player_color), color_choice, status, game, _btn_start()

    status = f"Played {player_san} for you. Opponent played {opponent_san}. Your turn."
    return _board_svg(game.board, player_color), color_choice, status, game, gr.update()


def suggest_move(
    color_choice: str,
    game: ChessGame | None,
) -> tuple[str, str, str, ChessGame | None]:
    """Sample a move from the engine without applying it."""
    if game is None:
        return "", color_choice, "Please select a model and start a game first.", None

    player_color = chess.WHITE if color_choice == "White" else chess.BLACK

    if game.board.is_game_over():
        return _board_svg(game.board, player_color), color_choice, f"Game over — {game.board.result()}. Start a new game.", game

    suggested = game.sample_move()
    suggested_san = game.board.san(suggested)
    return _board_svg(game.board, player_color), color_choice, f"HumbleMeister suggested: {suggested_san} ({suggested.uci()})", game


with gr.Blocks(title="HumbleMeister Chess") as demo:
    gr.Markdown("# HumbleMeister Chess\nPlay against a transformer chess engine.")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=list(MODEL_REGISTRY.keys()),
            label="Model",
            interactive=True,
        )
        color_radio = gr.Radio(
            choices=["White", "Black"],
            value="White",
            label="Your color",
        )
        start_btn = gr.Button("Start game", variant="primary")

    board_svg = gr.HTML(label="Board")
    status_box = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        move_input = gr.Textbox(
            label="Your move (UCI e.g. e2e4, or SAN e.g. e4, Nf3, O-O)",
            placeholder="e2e4",
            scale=4,
        )
        with gr.Column(scale=1, min_width=80):
            move_btn = gr.Button("Submit my move", variant="primary")
            play_for_me_btn = gr.Button("Play for me")
            suggest_btn = gr.Button("Help me, I need advice")

    # per-session state: color string and ChessGame instance
    color_state = gr.State("White")
    game_state = gr.State(None)

    start_btn.click(
        fn=start_or_surrender,
        inputs=[model_dropdown, color_radio, game_state],
        outputs=[board_svg, color_state, status_box, game_state, start_btn],
    )

    move_btn.click(
        fn=play_move,
        inputs=[color_state, move_input, game_state],
        outputs=[board_svg, color_state, status_box, game_state, start_btn],
    )

    move_input.submit(
        fn=play_move,
        inputs=[color_state, move_input, game_state],
        outputs=[board_svg, color_state, status_box, game_state, start_btn],
    )

    play_for_me_btn.click(
        fn=play_for_me,
        inputs=[color_state, game_state],
        outputs=[board_svg, color_state, status_box, game_state, start_btn],
    )

    suggest_btn.click(
        fn=suggest_move,
        inputs=[color_state, game_state],
        outputs=[board_svg, color_state, status_box, game_state],
    )


if __name__ == "__main__":
    demo.launch()
