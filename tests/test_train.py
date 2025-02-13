import io
import contextlib
import pytest
from src.train import main

def dummy_download_shakespeare(file_path="shakespeare_input.txt"):
    # Return a minimal text sample for testing purposes.
    return ("To be, or not to be: that is the question. " * 3).strip()

def test_train_runs_without_error(monkeypatch):
    # Monkey-patch download_shakespeare to use our dummy text.
    monkeypatch.setattr("src.train.download_shakespeare", dummy_download_shakespeare)
    
    captured_output = io.StringIO()
    # Redirect stdout to capture the printed output of the training process.
    with contextlib.redirect_stdout(captured_output):
        main()
    output = captured_output.getvalue()
    
    # Check that the training completes and the generated text is printed.
    assert "Generated Text:" in output