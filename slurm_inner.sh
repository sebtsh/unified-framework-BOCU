#!/bin/sh

ulimit -u 100000
ulimit -l unlimited
ulimit -d unlimited
ulimit -m unlimited
ulimit -v unlimited


python job_runner.py
