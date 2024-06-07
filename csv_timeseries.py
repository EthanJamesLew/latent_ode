"""
Adapter Class to run our benchmarks from this code
"""
import pandas as pd
import numpy as np
import sysidexpr.benchmark as sidbench
from benchmarks import benchmarks
import torch
import lib.utils as utils
from generate_timeseries import TimeSeries


def my_collate_fn(batch, args, device = torch.device("cpu"), data_type = "train"):
    """
    Expects a batch of M time series data in the form of np.array([times | states]) where
        - times is a (T,) length array
        - states is a (T, D) 
    Returns:
        combined_tt: The union of all time observations.
        combined_vals: (M, T, D) tensor containing the observed values.
        combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0].shape[-1]-1

    combined_tt, inverse_indices = torch.unique(torch.cat([torch.Tensor().new_tensor(ex[:, 0]).to(device) for ex in batch]), sorted=True, return_inverse=True)
    combined_tt = combined_tt.to(device)
    
    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)

    combined_labels = None
    N_labels = 1

    combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
    combined_labels = combined_labels.to(device = device)
	
    for b, res in enumerate(batch):
        tt = torch.Tensor().new_tensor(res[:, 0]).to(device)
        vals = torch.Tensor().new_tensor(res[:, 1:]).to(device)
        mask = torch.Tensor().new_tensor(np.ones(vals.shape)).to(device)

        indices = inverse_indices[offset:offset + len(tt)]
        offset += len(tt)

        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask

    #combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask, att_min = None, att_max = None)

    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / torch.max(combined_tt)
        
    data_dict = {
        "data": combined_vals, 
        "time_steps": combined_tt,
        "mask": combined_mask,
        #"labels": combined_labels
        }

    data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
    return data_dict


class BenchmarkTimeseries(TimeSeries):
    """
    load timeseries from a CSV file
    """
    def __init__(self, benchmark, data_len=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark = benchmark
        self.data_len = data_len
        self.trajectories = self.benchmark.load_trajectories()

    def get_times(self):
        data = self.trajectories['Train']
        ids = sorted(data._trajs.keys())
        for k in ids:
           t = data[k]
           times = t.times
        return torch.Tensor().new_tensor(times, device = self.device)

    @property
    def states(self):
        return self.benchmark.config.states
    
    @property
    def time_id(self):
        return self.benchmark.config.time
    
    @property
    def traj_id(self):
        return self.benchmark.config.traj
    

    def predictions_df(self, trajectories, time_steps):
        """
            Produce a predictions CSV from a pytorch Tensor
            @param trajectories: torch.Tensor [1, trajectory_idx, time_pts, states]

            Should look like

            header: <self.traj_id> | <self.time_id> | <self.states[0]> | ... | <self.states[n]> | Test | Train
            data: trajectories time points in 2D where a row is [self.traj_id[trajectory_idx], time_point, *states, True, False]
        """
        data = self.trajectories['Test']
        ids = sorted(data._trajs.keys())
        header = [self.traj_id, self.time_id] + self.states + ['Test', 'Train']
        data = []

        for traj_idx in range(trajectories.shape[1]):
            for time_point in range(trajectories.shape[2]):
                row = [ids[traj_idx], time_steps[time_point].numpy()] + trajectories[0, traj_idx, time_point, :].tolist() + [True, False]
                data.append(row)

        return pd.DataFrame(data, columns=header)

    def get_traj_list(self, key):
        data = self.trajectories[key]
        traj_list = []
        ids = sorted(data._trajs.keys())
        for k in ids:
            t = data[k]
            if self.data_len is not None:
                if len(t.times) < self.data_len:
                    continue
                traj = np.hstack((np.atleast_2d(t.times[:self.data_len]).T, t.states[:self.data_len]))
            else:
                traj = np.hstack((np.atleast_2d(t.times).T, t.states))
            #traj = np.expand_dims(traj[:,1:], 0)
            traj_list.append(traj)

        #traj_list = np.array(traj_list)
        #traj_list = torch.Tensor().new_tensor(traj_list, device = self.device)
        #traj_list = traj_list.squeeze(1)
        return traj_list

    def sample_traj(self, n_samples = 1):
        train = self.get_traj_list('Train')
        test = self.get_traj_list('Test')
        return train, test
