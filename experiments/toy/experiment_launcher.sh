rm outputs/*
rm images/*
for pid in $(ps -ea | grep "python2" | awk '{print $1}'); do kill $pid; done
python experiment_launcher.py
