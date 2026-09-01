#!/usr/bin/env bash
# Stream training progress as chat events: one line per STRIDE steps per run, plus any
# failure signature, plus completion. Exits once every run has finished or died.
#
# Usage: scripts/watch_training.sh <stride> <log> [<log> ...]
set -u
stride="$1"; shift
logs=("$@")
n_logs=${#logs[@]}
# Indexed (not associative) arrays: a file PATH used as an associative key breaks inside
# $(( )), where the subscript is evaluated as arithmetic and a path collapses to 0.
declare -a bucket prev_n prev_t fin
for ((k = 0; k < n_logs; k++)); do bucket[k]=-1; prev_n[k]=0; prev_t[k]=0; fin[k]=0; done

while :; do
  all_done=1
  for ((k = 0; k < n_logs; k++)); do
    (( fin[k] )) && continue
    f="${logs[k]}"; name=$(basename "$f" .log)
    [ -f "$f" ] || { all_done=0; continue; }

    # Failure signatures first -- silence must never look like success.
    err=$(grep -aoE "Traceback|out of memory|MemoryError|Killed|Segmentation fault|AssertionError|ValueError|KeyError" "$f" | tail -1)

    # tqdm writes with \r, so split on it and take the last progress token.
    prog=$(tr '\r' '\n' < "$f" | grep -aoE "[0-9]+/[0-9]+ \[" | tail -1 | tr -d ' [')
    n=${prog%%/*}; total=${prog##*/}

    if [ -n "$err" ] && ! pgrep -f "train.py" >/dev/null 2>&1; then
      echo "$name FAILED: $err"; fin[k]=1; continue
    fi
    if ! [[ ${n:-} =~ ^[0-9]+$ && ${total:-} =~ ^[0-9]+$ ]] || [ "$total" -le 0 ]; then
      all_done=0; continue
    fi

    b=$(( n / stride ))
    if (( b > bucket[k] )); then
      now=$SECONDS
      # Rate from the delta since the LAST emission, not since watcher start: the runs
      # may predate the watcher, and a cumulative average hides slowdowns.
      if (( prev_t[k] > 0 && now > prev_t[k] )); then
        rate=$(( (n - prev_n[k]) / (now - prev_t[k]) ))
        eta=$(( rate > 0 ? (total - n) / rate / 60 : -1 ))
        if (( eta >= 0 )); then
          echo "$name ${n}/${total} ($(( 100 * n / total ))%) ~${rate} it/s eta ${eta}m"
        else
          echo "$name ${n}/${total} ($(( 100 * n / total ))%) STALLED - no progress"
        fi
      else
        echo "$name ${n}/${total} ($(( 100 * n / total ))%) rate pending"
      fi
      bucket[k]=$b; prev_n[k]=$n; prev_t[k]=$now
    fi
    if (( n >= total )); then echo "$name COMPLETE (${n} steps)"; fin[k]=1; continue; fi
    all_done=0
  done
  (( all_done )) && break
  sleep 20
done
echo "all runs finished"
