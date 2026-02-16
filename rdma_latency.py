#!/usr/bin/env python3
"""
RDMA Latency Simulator: GPUDirect vs PCIe/NVLink/IB.
Multi-process sim of GPU-GPU data transfer latency.
"""

import argparse
import multiprocessing as mp
import time
import numpy as np

LATENCIES = {
    'pcie_gen5': 0.6e-6,  # 600ns
    'nvlink4': 0.05e-6,   # 50ns
    'ib_nic': 1.0e-6,     # 1us
    'gpudirect_rdma': 0.2e-6  # 200ns (RDMA bypass)
}

BANDWIDTHS = {
    'pcie_gen5': 64,  # GB/s
    'nvlink4': 900,
    'ib_nic': 400,
    'gpudirect_rdma': 400
}

def transfer(rank, size, payload_gb, transport):
    """Simulate GPU-GPU transfer."""
    latency = LATENCIES[transport]
    bw = BANDWIDTHS[transport]
    
    time_latency = latency
    time_bw = payload_gb / bw
    total = time_latency + time_bw
    
    time.sleep(total)  # Sim delay
    return total

def benchmark(transport, world_size, payload_gb):
    """Benchmark round-trip."""
    start = time.time()
    with mp.Pool(world_size) as pool:
        args = [(i, world_size, payload_gb, transport) for i in range(world_size)]
        latencies = pool.starmap(transfer, args)
    end = time.time()
    
    avg_latency = np.mean(latencies)
    throughput = (world_size * payload_gb * 2) / (end - start)  # GB/s bidirectional
    
    return avg_latency * 1e6, throughput  # us, GB/s

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', type=int, default=8)
    parser.add_argument('--payload', type=float, default=1.0)  # GB
    args = parser.parse_args()
    
    print(f"RDMA Sim: {args.world_size} GPUs, {args.payload}GB payload")
    
    for transport in LATENCIES:
        lat_us, tp_gbs = benchmark(transport, args.world_size, args.payload)
        print(f"{transport}: {lat_us:.0f}us latency, {tp_gbs:.1f} GB/s TP")
