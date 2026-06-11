#!/bin/bash
# 监控 train_metrics_live.txt，每出现新 epoch 行就写入 epoch_alerts.txt
LIVE="/root/autodl-tmp/真课设/语音信息处理课设/logs/train_metrics_live.txt"
ALERT="/root/autodl-tmp/真课设/语音信息处理课设/logs/epoch_alerts.txt"
touch "$ALERT"
declare -A seen
while pgrep -f 'python -u train_large.py' >/dev/null 2>&1; do
  while IFS= read -r line; do
    [[ "$line" =~ ^Epoch ]] || continue
    key=$(echo "$line" | awk '{print $2}')
    if [[ -z "${seen[$key]}" ]]; then
      seen[$key]=1
      ts=$(date '+%H:%M:%S')
      echo "[$ts] $line" >> "$ALERT"
    fi
  done < "$LIVE"
  sleep 15
done
ts=$(date '+%H:%M:%S')
echo "[$ts] TRAINING_FINISHED" >> "$ALERT"
