from functools import lru_cache

import ctypes
import numpy as np

Q = 3329
K = 2
KYBER_LIB = ctypes.cdll.LoadLibrary(f'libpqcrystals_kyber512_ref.so')

def compress_msg(x):
    return (((x << 1) + (Q >> 1)) // Q) & 1

class Poly(ctypes.Structure):
    _fields_ = [('coeffs', (256 * ctypes.c_int16))]

    def to_numpy(self):
        arr = np.ctypeslib.as_array(self.coeffs)

        return (arr % Q).astype(np.int32)

def unpack_sk(sk):
    s = (Poly * K)()

    KYBER_LIB.pqcrystals_kyber512_ref_polyvec_frombytes(s, sk)

    return s

def unpack_c(c):
    u = (Poly * K)()
    v = Poly()

    KYBER_LIB.pqcrystals_kyber512_ref_polyvec_decompress(u, c)
    # this is not correct for K = 4, it should be K * 352
    KYBER_LIB.pqcrystals_kyber512_ref_poly_decompress(ctypes.pointer(v), c[K * 320:])

    return (u, v)

@lru_cache(maxsize=100_000)
def extract_msg(sk, c, v_mask=0):
    s = unpack_sk(sk)
    u, v = unpack_c(c)

    z = Poly()
    for i in range(K):
        t = Poly()

        KYBER_LIB.pqcrystals_kyber512_ref_poly_ntt(ctypes.pointer(u[i]))
        KYBER_LIB.pqcrystals_kyber512_ref_poly_basemul_montgomery(ctypes.pointer(t), ctypes.pointer(u[i]), ctypes.pointer(s[i]))

        for j in range(256):
            z.coeffs[j] = (z.coeffs[j] + t.coeffs[j]) % Q

    KYBER_LIB.pqcrystals_kyber512_ref_poly_invntt_tomont(ctypes.pointer(z))

    v = v.to_numpy()

    if v_mask != 0:
        indices = [i for i in range(256) if (v_mask & (1 << (i & 0xf)))]
        v[indices] = (v[indices] + (Q >> 1)) % Q

    w = v - z.to_numpy()
    o = compress_msg(w % Q)

    o = (np.concat((v[-8:], o[:-8])) - o) % 2

    return np.squeeze(np.dot(o.reshape(-1, 32, 8), 2 ** np.arange(8)).astype(np.uint16))
