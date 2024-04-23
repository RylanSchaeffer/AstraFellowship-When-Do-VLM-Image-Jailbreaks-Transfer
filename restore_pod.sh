cp /workspace/bashrc ~/.bashrc
cp -r /workspace/.ssh/* /root/.ssh/
chmod 600 /root/.ssh/id_ed25519
eval "$(ssh-agent -s)"
ssh-add /root/.ssh/id_ed25519
source ~/.bashrc