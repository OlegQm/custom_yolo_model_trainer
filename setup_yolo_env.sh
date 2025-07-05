#!/bin/bash

set -e

echo "=== Cloning repo ==="
cd ~
git clone https://github.com/OlegQm/manipulator_project_datasets || echo "Repo manipulator_project_datasets already exists"
git clone https://github.com/OlegQm/custom_yolo_model_trainer || echo "Repo custom_yolo_model_trainer already exists"

echo "=== Update apt and install dependencies ==="
sudo apt update
sudo apt install -y python3.12 python3.12-venv
sudo apt update
sudo apt install -y libgl1
sudo apt install -y nvidia-driver-535 nvidia-utils-535

echo "=== Download the testing image ==="
cd ~/custom_yolo_model_trainer/ultralytics_custom/ultralytics/assets/
wget -nc https://ultralytics.com/images/bus.jpg

echo "=== Virtual environment setup ==="
cd ~/custom_yolo_model_trainer
python3.12 -m venv yolo_trainer
source yolo_trainer/bin/activate

echo "=== Install Ultralytics in editable-mode ==="
cd ~/custom_yolo_model_trainer/ultralytics_custom
pip install --upgrade pip
pip install -e .

echo "=== Generate SSH-key and enter it ==="
mkdir -p ~/.ssh
chmod 700 ~/.ssh

rm -f ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub

ssh-keygen -t ed25519 -C "olegqm@gmail.com" -f ~/.ssh/id_ed25519 -N ""
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

echo
echo "=== IMPORTANT: Copy that SSH-public key to GitHub (Settings -> SSH keys) ==="
cat ~/.ssh/id_ed25519.pub
echo

if ! grep -q "github.com" ~/.ssh/config 2>/dev/null; then
    echo -e "Host github.com\n\tIdentityFile ~/.ssh/id_ed25519\n" >> ~/.ssh/config
fi

echo "=== Setup cron-job for autopush ==="

CRON_PUSH="*/10 * * * * cd ~/custom_yolo_model_trainer && git add runs/ && git commit -m 'Trainer update' && GIT_SSH_COMMAND='ssh -i /home/ubuntu/.ssh/id_ed25519' git push >> ~/cron_git_push.log 2>&1"

(crontab -l 2>/dev/null | grep -v 'git add runs/' ; echo "$CRON_PUSH") | crontab -

echo
echo "=== Installation finished! ==="
echo "Check SSH key in your GitHub profile."
echo "Cron-job will now push changes from custom_yolo_model_trainer every 10 minutes."
echo

