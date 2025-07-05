#!/bin/bash

if ! command -v tmux >/dev/null 2>&1; then
    echo "Installing tmux..."
    sudo apt update
    sudo apt install -y tmux
fi

SESSION_NAME="yolo_train_session"

tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    tmux new-session -d -s $SESSION_NAME "cd ~/custom_yolo_model_trainer && python3 trainer.py"
    echo "tmux session $SESSION_NAME created and training started."
else
    echo "tmux session $SESSION_NAME already exists."
fi

echo "You can attach to the session with: tmux attach -t $SESSION_NAME"

