import gradio as gr
import pandas as pd
from solver import Solver

solver = Solver()


def get_next_word() -> tuple[str, pd.DataFrame]:
    """Get the next suggested word from the solver."""
    word = solver.next_guess()
    return word, to_dataframe()


def submit_guess(word: str, rank: int) -> tuple[str, pd.DataFrame]:
    """Record a guess and rank, then get the next suggestion."""
    if not word or not word.strip():
        return "", to_dataframe()
    word = word.strip().lower()
    solver.record(word, int(rank))
    next_word = solver.next_guess()
    return next_word, to_dataframe()


def add_manual_guess(word: str, rank: int, current_suggestion: str) -> tuple[pd.DataFrame, str]:
    """Manually add a word and rank without changing the current suggestion."""
    if not word or not word.strip():
        return to_dataframe(), current_suggestion
    solver.record(word.strip().lower(), int(rank))
    return to_dataframe(), current_suggestion


def reset() -> tuple[str, pd.DataFrame]:
    """Reset the solver for a new game."""
    solver.reset()
    first_word = solver.next_guess()
    return first_word, to_dataframe()


def to_dataframe() -> pd.DataFrame:
    """Convert solver state to a sorted dataframe for display."""
    state = solver.state()
    if not state:
        return pd.DataFrame(columns=["word", "rank"])
    return pd.DataFrame(state)


with gr.Blocks(title="Contexto Solver") as app:
    gr.Markdown("# Contexto Solver")
    gr.Markdown("Enter guesses and their ranks from Contexto. The solver will suggest the next word.")

    with gr.Row():
        with gr.Column(scale=1):
            suggestion = gr.Textbox(
                label="Next suggested word",
                interactive=False,
                value=""
            )
            next_btn = gr.Button("Get first word", variant="primary")

            gr.Markdown("### Add a guess")
            with gr.Row():
                word_input = gr.Textbox(label="Word", placeholder="e.g. water")
                rank_input = gr.Number(label="Rank", value=1000, precision=0)

            with gr.Row():
                add_btn = gr.Button("Add guess + get next word")
                manual_btn = gr.Button("Add guess only")

            reset_btn = gr.Button("Reset", variant="stop")

        with gr.Column(scale=1):
            table = gr.Dataframe(
                headers=["word", "rank"],
                datatype=["str", "number"],
                label="Guesses (sorted by rank)",
                interactive=False,
            )

    next_btn.click(
        fn=get_next_word,
        outputs=[suggestion, table]
    )

    add_btn.click(
        fn=submit_guess,
        inputs=[word_input, rank_input],
        outputs=[suggestion, table]
    )

    manual_btn.click(
        fn=add_manual_guess,
        inputs=[word_input, rank_input, suggestion],
        outputs=[table, suggestion]
    )

    reset_btn.click(
        fn=reset,
        outputs=[suggestion, table]
    )

if __name__ == "__main__":
    app.launch()