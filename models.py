import numpy as np
import matplotlib.pyplot as plt

from scalib.modeling import MultiLDA

# evaluate accuracy when multiple traces are combined to predict single byte
def multi_labels_eval(probs, labels, m):
    targets = labels[::m]
    labels = labels ^ np.repeat(targets, m)

    # shift probabilities, so each should predict first label from `m` labels
    shifted_probs = probs[np.arange(len(labels))[:, None], np.arange(256) ^ labels[:, None]]

    # group by `m` - multiple probabilities are combined together to get better precision
    grouped_probs = shifted_probs.reshape(-1, m, 256)

    # product in logspace as normal product requires higher precision
    probs = np.sum(np.log(grouped_probs + 1e-20), axis=1)

    predictions = np.argmax(probs, axis=1)

    sorted_indices = np.argsort(probs, axis=1)[:, ::-1]
    ranks = [np.where(sorted_indices[i] == targets[i])[0][0] for i in range(len(targets))]

    return np.mean(predictions == targets, axis=0), max(ranks)

class LdaModel:
    def __init__(self, pois, nc):
        self.multi_lda = MultiLDA(len(pois) * [nc], len(pois) * [1], pois)

    def __call__(self, traces):
        return self.multi_lda.predict_proba(traces)

    def train(self, data_loader):
        for traces, labels in data_loader:
            self.multi_lda.fit_u(traces, labels)

        self.multi_lda.solve()

    def eval(self, data_loader, m=256):
        accs, m_accs = [], [[] for _ in range(len(self.multi_lda.pois))]
        m_r, mm_r = 0, [0 for _ in range(len(self.multi_lda.pois))]

        for traces, labels in data_loader:
            probs = self(traces)

            predictions = np.argmax(probs, axis=2)
            accs.append(np.mean(predictions.T == labels, axis=0))

            sorted_indices = np.argsort(probs, axis=2)[:, ::-1]

            for i in range(len(probs)):
                ranks = [np.where(sorted_indices[i, j] == labels[j, i])[0][0] for j in range(len(labels))]

                m_r = max(m_r, max(ranks))

                acc, max_rank = multi_labels_eval(probs[i], labels[:, i], m)

                m_accs[i].append(acc)
                mm_r[i] = max(mm_r[i], max_rank)

        def fmt(arr):
            return ', '.join(map(lambda x: f'{float(x):.2g}', arr))

        print(f'Acc: {fmt(sum(accs) / len(accs))}')
        print(f'Multi acc: {fmt([sum(m_accs[i]) / len(m_accs[i]) for i in range(len(m_accs))])}')
        print(f'Max rank: {m_r}, Multi max rank: {fmt(mm_r)}')


        return m_accs, mm_r

    # combines multiple traces into single label prediction, `offsets` represents known change in guessed byte
    def predict(self, traces, offsets, label):
        predicted_bytes = []

        for probs in self(traces):
            shifted_probs = probs[np.arange(len(offsets))[:, None], np.arange(256) ^ offsets]

            final_probs = np.sum(np.log(shifted_probs + 1e-20), axis=0)

            predicted_bytes.append(int(np.argmax(final_probs, axis=0)))

        return predicted_bytes
