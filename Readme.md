# Message Recovery Attack on ML-KEM
This repository provides the implementation and datasets for a message recovery attack against a [hardware implementation of ML-KEM](https://gitlab.com/brno-axe/pqc/diky).

The power traces were measured on [Sakura-X board](http://satoh.cs.uec.ac.jp/SAKURA/hardware/SAKURA-X.html) processing the key decapsulation phase and cut to contain only the vulnerable regions as described in paper.

## Setup
The attack verification script requires the reference implementation of Kyber to handle cryptographic primitives.
```bash
# Clone the reference implementation
git clone https://github.com/pq-crystals/kyber
cd kyber/ref

# Build the shared library
make shared
```

Ensure that libpqcrystals_kyber512_ref.so is added to your system's library path.

Install the necessary dependencies.
```bash
pip install -r requirements.txt
```

## Execute the attack

To execute the attack run. 
```bash
python main.py
```
