1. [x] parallize using mpi.
2. [ ] support gpu.
3. [ ] logger to support tag tensorboard tabular.
4. [ ] check data shape data value at initial iter.
5. [ ] other important metrics.
6. [ ] visualize.
7. [ ] add critic.
8. [ ] GAE.

ps -ef | grep trainer.py | grep -v grep | awk '{print $2}' | xargs kill -9