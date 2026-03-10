import numpy as np
import matplotlib.pyplot as plt

from scalib.metrics import SNR

class TraceDataset:
    def __init__(self, traces, labels_gen, subtract_mean=False):
        self.traces = traces
        self.labels_gen = labels_gen
        self.subtract_mean = subtract_mean

    def __getitem__(self, idx):
        traces, labels = (self.traces[idx], self.labels_gen(idx))

        if self.subtract_mean:
            traces = traces - np.mean(traces, axis=1)[-1, None]

        return traces.astype(np.int16), labels

class TraceDataLoader:
    def __init__(self, dataset, samples, batch_size, offset=0, progress_f=None):
        self.dataset = dataset
        self.samples = samples
        self.batch_size = batch_size
        self.offset = offset
        self.progress_f = progress_f

        self.current_idx = 0

    def with_label(self, label):
        return TraceDataLoader(self.dataset, self.samples, self.batch_size, self.offset, progress_f(label))

    def __len__(self):
        return (self.samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self.current_idx = 0

        return self

    def __next__(self):
        if self.current_idx >= self.samples:
            raise StopIteration

        traces, labels = self.dataset[self.current_idx + self.offset: self.current_idx + self.offset + self.batch_size]
        self.current_idx += self.batch_size

        if self.progress_f != None:
            self.progress_f(self.current_idx / self.samples)

        return traces, labels

def progress_f(title):
    def f(x):
        print(f'\r{title}: {100 * x:.1f}%', end='\n' if x == 1 else '', flush=True)

    return f

def find_pois(data_loader, nc, n_pois, plot=False):
    snr = SNR(nc)

    for traces, labels in data_loader:
        snr.fit_u(traces, labels)

    snr_res = snr.get_snr()

    pois = []
    for i, snr in enumerate(snr_res):
        if i == 0:
            pois.append([0])
            continue

        snr = np.nan_to_num(snr)
        peaks = np.argsort(snr)[-n_pois:]
        pois.append(peaks)

    if plot:
        plt.figure(figsize=(15, 7.5))
        plt.title('SNR')

        for i, snr in enumerate(snr_res):
            plt.plot(snr, label=i)
            plt.plot(pois[i], snr[pois[i]], 'ro')

        plt.legend(loc='upper right')
        plt.show()

    return pois
