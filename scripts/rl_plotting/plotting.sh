#!/bin/bash

grep "Loss" $1 > $2.loss
grep "consensus" $1 > $2.consensus
grep "simtime" $1 > $2.simtime
grep "Wallclock" $1 > $2.wallclock

cat <(echo ", Iteration, Loss") $2.loss > $2.loss
