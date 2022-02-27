1. parallize using mpi.
2. support gpu.
2. logger to support tag tensorboard tabular.
2. check data shape data value at initial iter.
3. other important metrics.
4. visualize.
ps -ef | grep demo1.py | grep -v grep | awk '{print $2}' | xargs kill -9