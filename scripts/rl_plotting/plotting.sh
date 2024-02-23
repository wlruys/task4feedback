#!/bin/bash

grep "Loss" $1 > $2.loss.old
grep "consensus" $1 > $2.consensus
grep "simtime" $1 > $2.simtime.old
grep "Wallclock" $1 > $2.wallclock.old

cat <(echo ", Iterations, Loss") $2.loss.old > $2.loss
cat <(echo ", Iterations, SimTime") $2.simtime.old > $2.simtime
cat <(echo ", Iterations, WallClock") $2.wallclock.old > $2.wallclock

SCRIPT_PATH=/oden/hlee/hlee/workspace/rl/task4feedback/scripts/rl_plotting
Rscript $SCRIPT_PATH/simtime.R $2.wallclock $2.simtime 
Rscript $SCRIPT_PATH/loss.R $2.loss
