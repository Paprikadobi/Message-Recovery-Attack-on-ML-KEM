import h5py
import numpy as np

from kyber import *
from models import *
from utils import *

def extract_msg_for_kem_dec(inputs, n):
    def f(idx):
        data = inputs[idx]

        labels = []

        for x in data:
            labels.append(extract_msg(bytes(x[:1632]), bytes(x[1632:2400]))[:n])

        return np.array(labels)

    return f

def extract_msg_for_kem_dec_attack(inputs, n):
    def f(idx):
        labels = []

        for i in range(idx.start, idx.stop):
            labels.append(extract_msg(bytes(inputs[i >> 8][:1632]), bytes(inputs[i >> 8][1632:2400]), i & 0xff)[:n])

        return np.array(labels)

    return f

with h5py.File('datasets/kem_dec_unprotected_8.h5') as file:
    COEFFS = 32

    dataset = TraceDataset(file['traces'], extract_msg_for_kem_dec(file['inputs'], COEFFS))

    train_data_loader = TraceDataLoader(dataset, 20_000, 1_000)

    # pois = find_pois(train_data_loader.with_label('Finding PoIs'), 256, 400)

    pois = [
            [x + 67 * i - (i + 1) // 3 for x in range(233, 634)] +
            [x + 67 * i - (i + 1) // 3 for x in range(3_367, 3_500)]
            # [x + 67 * i - (i + 1) // 3 for x in range(367, 434)] +
            # [x + 67 * i - (i + 1) // 3 for x in range(500, 567)] +
            # [x + 67 * i - (i + 1) // 3 for x in range(3_367, 3_500)]
        for i in range(COEFFS)]

    lda_model = LdaModel(pois, 256)

    lda_model.train(train_data_loader.with_label('Training LDA'))

    with h5py.File('datasets/kem_dec_unprotected_16_attack.h5') as file_attack:
        dataset_attack = TraceDataset(file_attack['traces'], extract_msg_for_kem_dec_attack(file_attack['inputs'], COEFFS))
        lda_model.eval(TraceDataLoader(dataset_attack, 200 * 256, 1_024).with_label('Evaluating LDA'))

    with h5py.File('datasets/kem_dec_unprotected_4_attack.h5') as file_attack:
        dataset_attack = TraceDataset(file_attack['traces'], extract_msg_for_kem_dec_attack(file_attack['inputs'], COEFFS))
        lda_model.eval(TraceDataLoader(dataset_attack, 200 * 256, 1_024).with_label('Evaluating LDA'))
