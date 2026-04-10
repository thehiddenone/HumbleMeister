from __future__ import annotations

import chess
import chess.svg
import gradio as gr
from huggingface_hub import snapshot_download

from humblemeister import ChessEngine

# Add HF repos here
MODEL_REGISTRY: dict[str, str] = {
    # "Giant v1": "StanStanStan3141592/hmeister-giant-v1",
}

_engines: dict[str, ChessEngine] = {}


def _resolve_path(model_name: str) -> str:
    """Download model from HF Hub if needed and return local cache path."""
    repo_id = MODEL_REGISTRY[model_name]
    return snapshot_download(repo_id=repo_id)


def _get_engine(model_name: str) -> ChessEngine:
    if model_name not in _engines:
        path = _resolve_path(model_name)
        _engines[model_name] = ChessEngine.load(path, device="cpu")
    return _engines[model_name]


def _board_svg(board: chess.Board, player_color: chess.Color) -> str:
    lastmove = board.peek() if board.move_stack else None
    flipped = player_color == chess.BLACK
    return chess.svg.board(board, lastmove=lastmove, size=400, flipped=flipped)  # type: ignore


def start_game(model_name: str, color_choice: str) -> tuple[str, str, str]:
    """Called when the player picks a model and color."""
    if not model_name:
        return "", "", "Please select a model first."

    engine = _get_engine(model_name)
    player_color = chess.WHITE if color_choice == "White" else chess.BLACK
    engine.start_game(player_color)

    status = "Your turn." if player_color == chess.WHITE else f"Model played. Your turn."
    return _board_svg(engine.board, player_color), color_choice, status


def play_move(
    model_name: str, color_choice: str, san_input: str
) -> tuple[str, str, str]:
    """Called when the player submits a move."""
    if not model_name:
        return "", color_choice, "Please select a model and start a game first."

    engine = _get_engine(model_name)
    player_color = chess.WHITE if color_choice == "White" else chess.BLACK

    if engine.board.is_game_over():
        return _board_svg(engine.board, player_color), color_choice, f"Game over — {engine.board.result()}. Start a new game."

    # parse SAN
    try:
        move = engine.board.parse_san(san_input.strip())
    except ValueError:
        return _board_svg(engine.board, player_color), color_choice, f"Invalid move: '{san_input}'. Please use SAN notation (e.g. e4, Nf3, O-O)."

    # apply player move
    engine._apply_move(move)

    if engine.board.is_game_over():
        return _board_svg(engine.board, player_color), color_choice, f"Game over — {engine.board.result()}"

    # model responds
    model_move = engine._sample_move()
    engine._apply_move(model_move)
    model_san = engine.board.san(model_move) if engine.board.move_stack else model_move.uci()

    if engine.board.is_game_over():
        status = f"Model played {model_move.uci()}. Game over — {engine.board.result()}"
    else:
        status = f"Model played {model_move.uci()}. Your turn."

    return _board_svg(engine.board, player_color), color_choice, status


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
            label="Your move (SAN notation, e.g. e4, Nf3, O-O)",
            placeholder="e4",
            scale=4,
        )
        move_btn = gr.Button("Play", variant="primary", scale=1)

    # hidden state to carry color across calls
    color_state = gr.State("White")

    start_btn.click(
        fn=start_game,
        inputs=[model_dropdown, color_radio],
        outputs=[board_svg, color_state, status_box],
    )

    move_btn.click(
        fn=play_move,
        inputs=[model_dropdown, color_state, move_input],
        outputs=[board_svg, color_state, status_box],
    )

    move_input.submit(
        fn=play_move,
        inputs=[model_dropdown, color_state, move_input],
        outputs=[board_svg, color_state, status_box],
    )


if __name__ == "__main__":
    demo.launch()
