import h5py
import numpy as np

from kyber import *
from models import *
from utils import *

import matplotlib.ticker as mticker

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

    pois = find_pois(train_data_loader.with_label('Finding PoIs'), 256, 400)

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
        m_accs16, mm_r16 = lda_model.eval(TraceDataLoader(dataset_attack, 200 * 256, 1_024).with_label('Evaluating LDA'))

    with h5py.File('datasets/kem_dec_unprotected_8_attack.h5') as file_attack:
        dataset_attack = TraceDataset(file_attack['traces'], extract_msg_for_kem_dec_attack(file_attack['inputs'], COEFFS))
        m_accs8, mm_r8 = lda_model.eval(TraceDataLoader(dataset_attack, 200 * 256, 1_024).with_label('Evaluating LDA'))

    with h5py.File('datasets/kem_dec_unprotected_4_attack.h5') as file_attack:
        dataset_attack = TraceDataset(file_attack['traces'], extract_msg_for_kem_dec_attack(file_attack['inputs'], COEFFS))
        m_accs4, mm_r4 = lda_model.eval(TraceDataLoader(dataset_attack, 200 * 256, 1_024).with_label('Evaluating LDA'))

    # --- Data Preparation ---
    x = range(1, 32)

    def get_acc(m_accs_list):
        return [sum(m_accs_list[i]) / len(m_accs_list[i]) for i in range(len(m_accs_list))][1:]

    # Calculate metrics for all three datasets
    acc16, rank16 = get_acc(m_accs16), mm_r16[1:]
    acc8,  rank8  = get_acc(m_accs8),  mm_r8[1:]
    acc4,  rank4  = get_acc(m_accs4),  mm_r4[1:]

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(9, 6), dpi=100)
    ax2 = ax1.twinx()

    # 1. Plot Accuracy (Left Axis) - Blue/Teal Gradient
    # Solid lines with distinct markers for different averaging levels
    l1, = ax1.plot(x, acc16, label='Acc (16 avg)', color='#003f5c', lw=2, marker='o', ms=4)
    l2, = ax1.plot(x, acc8,  label='Acc (8 avg)',  color='#2f4b7c', lw=2, marker='^', ms=4)
    l3, = ax1.plot(x, acc4,  label='Acc (4 avg)',  color='#665191', lw=2, marker='s', ms=4, alpha=0.7)

    ax1.set_xlabel('Message Byte Index')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_xlim(1, 31)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # 2. Plot Rank (Right Axis) - Orange/Red Gradient
    # Using different line styles (Dashed, Dash-Dot, Dotted) to separate them from Accuracy
    l4, = ax2.plot(x, rank16, label='Rank (16 avg)', color='#f95d6a', ls='--', lw=1.5)
    l5, = ax2.plot(x, rank8,  label='Rank (8 avg)',  color='#ff7c43', ls='-.', lw=1.5)
    l6, = ax2.plot(x, rank4,  label='Rank (4 avg)',  color='#ffa600', ls=':',  lw=1.5)

    ax2.set_ylabel('Rank')
    ax2.set_ylim(-0.1, 4)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # 3. Unified Legend
    # Organized in 3 columns so that Acc and Rank for each N-value align vertically
    lines = [l1, l2, l3, l4, l5, l6]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=3, frameon=False, fontsize=9)

    plt.xticks(range(1, 32, 2))

    fig.tight_layout()

    # Save for your report
    plt.savefig('template_attack_results.pdf', bbox_inches='tight', dpi=600)
    plt.show()
