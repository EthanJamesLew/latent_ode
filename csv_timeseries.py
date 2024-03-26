"""
Adapter Class to run our benchmarks from this code
"""
import pandas as pd
import numpy as np
import sysidexpr.benchmark as sidbench
from benchmarks import benchmarks
import torch

from generate_timeseries import TimeSeries

class BenchmarkTimeseries(TimeSeries):
    """
    load timeseries from a CSV file
    """
    def __init__(self, benchmark, data_len=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark = benchmark
        self.data_len = data_len


    def get_times(self):
        trajectories = self.benchmark.load_trajectories()
        data = trajectories['Train']
        for t in data:
           times = t.times
        return torch.Tensor().new_tensor(times, device = self.device)

    def get_traj_list(self, key):
        trajectories = self.benchmark.load_trajectories()
        data = trajectories[key]
        traj_list = []
        for t in data:
            if self.data_len is not None:
                if len(t.times) < self.data_len:
                    continue
                traj = np.hstack((np.atleast_2d(t.times[:self.data_len]).T, t.states[:self.data_len]))
            else:
                traj = np.hstack((np.atleast_2d(t.times).T, t.states))
            traj = np.expand_dims(traj[:,1:], 0)
            traj_list.append(traj)

        traj_list = np.array(traj_list)
        traj_list = torch.Tensor().new_tensor(traj_list, device = self.device)
        traj_list = traj_list.squeeze(1)
        return traj_list

    def sample_traj(self, n_samples = 1):
        train = self.get_traj_list('Train')
        test = self.get_traj_list('Test')
        return train, test
