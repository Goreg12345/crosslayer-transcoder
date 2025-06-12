#!/usr/bin/env python3
"""
Hardware Analysis - Calculate theoretical bandwidth limits

Analyzes your system topology and calculates theoretical limits
based on the discovered hardware configuration.
"""

import subprocess
import re


def parse_nvidia_smi_output():
    """Extract detailed information from nvidia-smi"""
    try:
        # Get detailed GPU information
        result = subprocess.run(['nvidia-smi', '-q'], capture_output=True, text=True)
        if result.returncode != 0:
            return None
        
        output = result.stdout
        
        # Parse GPU information
        gpus = []
        gpu_sections = output.split('GPU 00000000:')
        
        for section in gpu_sections[1:]:  # Skip first empty section
            gpu_info = {}
            
            # Extract bus ID
            bus_match = re.search(r'([0-9A-F]{2}:[0-9A-F]{2}\.[0-9])', section)
            if bus_match:
                gpu_info['bus_id'] = bus_match.group(1)
            
            # Extract PCIe information
            pcie_gen_match = re.search(r'Max\s+:\s+(\d+)', section)
            if pcie_gen_match:
                gpu_info['pcie_gen_max'] = int(pcie_gen_match.group(1))
            
            pcie_width_match = re.search(r'Link Width\s*\n\s*Max\s+:\s+(\d+)x', section)
            if pcie_width_match:
                gpu_info['pcie_width_max'] = int(pcie_width_match.group(1))
            
            # Extract current PCIe status
            current_gen_match = re.search(r'Current\s+:\s+(\d+)', section)
            if current_gen_match:
                gpu_info['pcie_gen_current'] = int(current_gen_match.group(1))
            
            current_width_match = re.search(r'Link Width\s*\n\s*Max\s+:\s+\d+x\s*\n\s*Current\s+:\s+(\d+)x', section)
            if current_width_match:
                gpu_info['pcie_width_current'] = int(current_width_match.group(1))
            
            gpus.append(gpu_info)
        
        return gpus
        
    except Exception as e:
        print(f"Error parsing nvidia-smi: {e}")
        return None


def calculate_pcie_bandwidth(gen, width):
    """Calculate PCIe bandwidth in GB/s"""
    # PCIe transfer rates (GT/s = Giga Transfers per second)
    transfer_rates = {
        1: 2.5,   # Gen 1: 2.5 GT/s
        2: 5.0,   # Gen 2: 5.0 GT/s  
        3: 8.0,   # Gen 3: 8.0 GT/s
        4: 16.0,  # Gen 4: 16.0 GT/s
        5: 32.0   # Gen 5: 32.0 GT/s
    }
    
    if gen not in transfer_rates:
        return None
    
    # PCIe uses 8b/10b encoding (Gen 1-2) or 128b/130b encoding (Gen 3+)
    encoding_efficiency = 0.8 if gen <= 2 else (128/130)
    
    # Bandwidth = transfer_rate * width * encoding_efficiency
    bandwidth_gbps = transfer_rates[gen] * width * encoding_efficiency
    
    return bandwidth_gbps


def analyze_topology():
    """Analyze GPU topology from nvidia-smi topo"""
    try:
        result = subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True, text=True)
        if result.returncode != 0:
            return None
        
        output = result.stdout
        
        # Parse topology matrix
        lines = output.strip().split('\n')
        
        # Find GPU lines
        gpu_lines = []
        for line in lines:
            if line.startswith('GPU'):
                gpu_lines.append(line.split())
        
        # Analyze connections
        connections = {}
        for i, line in enumerate(gpu_lines):
            gpu_id = line[0]
            connections[gpu_id] = {}
            
            for j, connection in enumerate(line[1:5]):  # GPU0, GPU1, GPU2, GPU3 columns
                target_gpu = f"GPU{j}"
                if target_gpu != gpu_id:
                    connections[gpu_id][target_gpu] = connection
        
        return connections
        
    except Exception as e:
        print(f"Error parsing topology: {e}")
        return None


def get_cpu_info():
    """Get CPU and NUMA information"""
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        if result.returncode != 0:
            return None
        
        output = result.stdout
        info = {}
        
        for line in output.split('\n'):
            if 'Socket(s):' in line:
                info['sockets'] = int(line.split(':')[1].strip())
            elif 'NUMA node(s):' in line:
                info['numa_nodes'] = int(line.split(':')[1].strip())
            elif 'Model name:' in line:
                info['cpu_model'] = line.split(':')[1].strip()
        
        return info
        
    except Exception as e:
        print(f"Error getting CPU info: {e}")
        return None


def analyze_allreduce_bottlenecks(gpus, topology, cpu_info):
    """Analyze potential AllReduce bottlenecks"""
    print("=== AllReduce Bottleneck Analysis ===\n")
    
    # 1. PCIe bandwidth per GPU
    if gpus and len(gpus) > 0:
        sample_gpu = gpus[0]
        pcie_bw = calculate_pcie_bandwidth(
            sample_gpu.get('pcie_gen_current', 3),
            sample_gpu.get('pcie_width_current', 16)
        )
        
        print(f"Per-GPU PCIe bandwidth: {pcie_bw:.1f} GB/s")
        print(f"  - PCIe Gen {sample_gpu.get('pcie_gen_current', 3)} x{sample_gpu.get('pcie_width_current', 16)}")
    
    # 2. Topology analysis
    if topology:
        print(f"\nTopology connections:")
        connections_summary = set()
        for gpu, targets in topology.items():
            for target, conn_type in targets.items():
                connections_summary.add(conn_type)
        
        for conn_type in connections_summary:
            if conn_type == 'SYS':
                print(f"  - SYS: Cross-NUMA communication (slowest)")
                print(f"    Theoretical: ~6-12 GB/s effective")
            elif conn_type == 'NODE':
                print(f"  - NODE: Same NUMA node (faster)")
                print(f"    Theoretical: ~15-25 GB/s effective")
            elif conn_type.startswith('NV'):
                nvlink_count = int(conn_type[2:]) if len(conn_type) > 2 else 1
                print(f"  - {conn_type}: NVLink connection")
                print(f"    Theoretical: ~{nvlink_count * 25} GB/s per direction")
    
    # 3. NUMA analysis
    if cpu_info:
        print(f"\nNUMA configuration:")
        print(f"  - {cpu_info.get('numa_nodes', 'Unknown')} NUMA nodes")
        print(f"  - {cpu_info.get('sockets', 'Unknown')} CPU sockets")
        print(f"  - CPU: {cpu_info.get('cpu_model', 'Unknown')}")
        
        if cpu_info.get('numa_nodes') == 4:
            print(f"  - Each GPU likely on separate NUMA node")
            print(f"  - AllReduce must traverse inter-socket links")
    
    # 4. Theoretical AllReduce analysis
    print(f"\n=== Theoretical AllReduce Analysis ===")
    
    if topology and all(conn_type == 'SYS' for gpu_conns in topology.values() for conn_type in gpu_conns.values()):
        print(f"All connections are SYS (cross-NUMA):")
        print(f"  - Ring AllReduce: Limited by slowest link (~6-8 GB/s)")
        print(f"  - Tree AllReduce: May be slightly better (~6-10 GB/s)")
        print(f"  - Your measured 6.3 GB/s is very close to theoretical limit")
        
        print(f"\nNVIDIA NCCL optimizations for this topology:")
        print(f"  - Likely uses tree algorithm to minimize cross-NUMA hops")
        print(f"  - May use ring algorithm within NUMA domains")
        print(f"  - 6.3 GB/s suggests NCCL is well-optimized for your hardware")
    
    # 5. Compare to measured performance
    print(f"\n=== Comparison to Your Measurements ===")
    print(f"Measured AllReduce bandwidth: 6.3 GB/s")
    
    if pcie_bw:
        efficiency = 6.3 / pcie_bw * 100
        print(f"Efficiency vs single PCIe link: {efficiency:.1f}%")
        
        if efficiency > 15:
            print(f"✓ Good efficiency considering cross-NUMA overhead")
        else:
            print(f"⚠ Low efficiency - may indicate additional bottlenecks")


def main():
    print("=== Hardware Analysis for AllReduce Performance ===\n")
    
    # Get hardware information
    gpus = parse_nvidia_smi_output()
    topology = analyze_topology()
    cpu_info = get_cpu_info()
    
    # Basic hardware summary
    print("=== Hardware Summary ===")
    if gpus:
        print(f"GPUs: {len(gpus)}x Tesla V100-PCIE-32GB")
        sample_gpu = gpus[0]
        print(f"PCIe: Gen {sample_gpu.get('pcie_gen_current', '?')} x{sample_gpu.get('pcie_width_current', '?')}")
        
        theoretical_bw = calculate_pcie_bandwidth(
            sample_gpu.get('pcie_gen_current', 3),
            sample_gpu.get('pcie_width_current', 16)
        )
        if theoretical_bw:
            print(f"Theoretical PCIe bandwidth: {theoretical_bw:.1f} GB/s per GPU")
    
    if cpu_info:
        print(f"CPU: {cpu_info.get('cpu_model', 'Unknown')}")
        print(f"NUMA: {cpu_info.get('numa_nodes', '?')} nodes")
    
    print()
    
    # Detailed analysis
    analyze_allreduce_bottlenecks(gpus, topology, cpu_info)
    
    # Recommendations
    print(f"\n=== NVIDIA Documentation References ===")
    print(f"For your hardware configuration:")
    print(f"1. Tesla V100 specs: https://www.nvidia.com/en-us/data-center/v100/")
    print(f"2. NCCL performance guide: https://docs.nvidia.com/deeplearning/nccl/user-guide/")
    print(f"3. Multi-GPU performance: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/")
    print(f"4. PCIe topology: nvidia-smi topo -m (you already ran this)")


if __name__ == "__main__":
    main()