# Data Loader

High-performance activation data loader for neural network interpretability research. Provides streaming access to transformer activations with shared memory for efficient multiprocessing.

## Quick Start

### Basic Usage

```python
from data import actvs_loader_from_test_config

# Create a test data loader (small buffer, fast startup)
loader = actvs_loader_from_test_config(batch_size=1000)

# Start using immediately - no manual setup needed!
for batch in loader:
    # batch shape: [batch_size, n_in_out, n_layers, activation_dim]
    # For GPT-2: [1000, 2, 12, 768]
    print(f"Batch shape: {batch.shape}")
    break
```

### Production Usage

```python
from data import actvs_loader_from_production_config

# Create a production data loader (large buffer, high throughput)
loader = actvs_loader_from_production_config(
    batch_size=5000,
    buffer_size=1_000_000,  # 1M samples
    generation_batch_size=64
)

# Use in training loop
for batch in loader:
    # High-throughput streaming data
    activations = batch  # [5000, 2, 12, 768]
    # ... train your model
```

### Custom Configuration

```python
from data import actvs_loader_from_config, DataLoaderConfig

# Create custom config
config = DataLoaderConfig(
    buffer_size=500_000,
    model_name="openai-community/gpt2",
    dataset_name="Skylion007/openwebtext",
    generation_batch_size=32,
    activation_dim=768,
    n_layers=12
)

loader = actvs_loader_from_config(config, batch_size=2000)
```

### Context Manager Usage

```python
from data import actvs_loader_from_production_config

# Automatic cleanup
with actvs_loader_from_production_config(batch_size=1000) as loader:
    for i, batch in enumerate(loader):
        if i >= 10:  # Process 10 batches
            break
# Process and resources automatically cleaned up
```

## Architecture

The data loader uses a modular architecture with clean separation of responsibilities:

### Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Factory       │───▶│ DataGenerator    │───▶│ SharedMemory    │
│   (Top-level)   │    │ Process          │    │ Buffer          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌────────▼────────┐             │
         │              │ DataGeneration  │             │
         │              │ Loop            │             │
         │              └─────────────────┘             │
         │                       │                      │
         └──────────────────┐    │    ┌─────────────────▼──────┐
                            │    │    │ SharedMemoryDataLoader │
                    ┌───────▼────▼────▼─────────┐              │
                    │ Component Ecosystem       │              │
                    └───────────────────────────┘              │
                                                               │
                    ┌──────────────────────────────────────────▼───┐
                    │           Training Script                     │
                    │   for batch in loader:                       │
                    │       # Use activations                      │
                    └───────────────────────────────────────────────┘
```

### Core Components

#### 1. **Factory Pattern** (`factory.py`)
**Responsibility**: Top-level orchestration and dependency injection
- `actvs_loader_from_config()` - Main factory function
- `actvs_loader_from_test_config()` - Quick test setup  
- `actvs_loader_from_production_config()` - Production setup
- **Key Feature**: All IO happens at the top level, no hidden operations

#### 2. **Data Generator Process** (`data_generator.py`)
**Responsibility**: Process lifecycle management
- Multiprocessing coordination
- Component assembly and dependency injection
- Resource cleanup and termination
- **Key Feature**: Focused only on process management, delegates work to specialists

#### 3. **Generation Loop** (`generation_loop.py`)
**Responsibility**: Engineering orchestration and optimization
- Device management (CPU/GPU switching based on buffer load)
- Buffer monitoring and decision making
- Performance optimization
- Coordinates between different data sources
- **Key Feature**: Contains all the engineering smarts

#### 4. **Activation Sources** (`activation_sources.py`)
**Responsibility**: Data source abstraction

**`ActivationComputer`** - Pure computation
- Takes model + tokens → returns activations
- Zero dependencies, easily testable
- Extracted exact activation computation logic

**`DiskActivationSource`** - File-based data
- Reads pre-computed activations from HDF5 files
- Sequential access with position tracking
- Fallback data source for fast buffer filling

#### 5. **Process Monitor** (`process_monitor.py`)
**Responsibility**: Logging, metrics, and dashboard
- Real-time CLI dashboard with buffer stats
- Performance metrics tracking
- Error logging and progress reporting
- **Key Feature**: All IO separated from business logic

#### 6. **Shared Memory Buffer** (`shared_memory.py`)
**Responsibility**: High-performance inter-process data sharing
- Named shared memory segments for fast process startup
- Custom pickling/unpickling for multiprocessing efficiency
- Validity tracking for partial data  
- Lock-safe tensor operations
- **Key Feature**: Optimized for high-throughput access with millisecond process startup

#### 7. **Data Loader Interface** (`dataset.py`)
**Responsibility**: PyTorch-compatible interface
- Iterator protocol implementation
- Batch management
- Context manager support
- **Key Feature**: Clean separation from process management

### Data Flow

```
1. Factory creates components with dependency injection
   ├── Models created in process (avoid pickle issues)
   ├── Dataset and tokenization setup
   ├── Shared memory buffer allocation
   └── Component wiring

2. DataGeneratorProcess starts background generation
   ├── DataGenerationLoop monitors buffer state
   ├── Chooses optimal device (CPU/GPU) based on load
   ├── ActivationComputer generates fresh data
   └── DiskActivationSource provides fallback data

3. SharedMemoryDataLoader provides clean interface
   ├── Implements PyTorch DataLoader protocol
   ├── Batch retrieval from shared buffer
   └── Statistics and monitoring

4. ProcessMonitor provides visibility
   ├── Real-time dashboard: "Buffer: 45,231/1,000,000 (4.5%) | Rate: 1,234/s"
   ├── Performance metrics
   └── Error reporting
```

### Key Design Principles

1. **Single Responsibility**: Each class has one clear job
2. **Dependency Injection**: All dependencies passed explicitly
3. **IO at Top Level**: No hidden file operations or model loading
4. **Process Boundary Respect**: Avoid pickle issues with complex objects
5. **Performance First**: Optimized for high-throughput training
6. **Clean Interfaces**: Easy to test, mock, and extend

### Performance Features

- **Fast Process Startup**: Custom pickling ensures `start()` returns in milliseconds, not minutes
- **Adaptive Device Selection**: Automatically switches between CPU/GPU based on buffer pressure
- **Shared Memory**: Zero-copy data sharing between processes using named shared memory segments
- **Continuous Generation**: Background process keeps buffer filled
- **Memory Optimization**: Efficient tensor storage and cleanup
- **Real-time Monitoring**: Live dashboard shows throughput and buffer status

### Configuration

The system uses a hierarchical configuration approach:

```python
DataLoaderConfig(
    # Buffer settings
    buffer_size=1_000_000,     # Number of samples in shared memory
    activation_dim=768,         # GPT-2 activation dimension
    n_layers=12,               # Number of transformer layers
    
    # Model settings  
    model_name="openai-community/gpt2",
    model_dtype=torch.float32,
    
    # Dataset settings
    dataset_name="Skylion007/openwebtext",
    max_sequence_length=1024,
    generation_batch_size=32,
    
    # Performance settings
    refresh_interval=0.1,       # Buffer check frequency
    max_batch_size=500_000,     # Upper limit for batch sizes
)
```

This architecture enables high-performance activation streaming while maintaining clean, testable, and maintainable code.

## Technical Implementation Details

### Fast Process Startup

The data loader solves a critical performance issue with large shared memory buffers. Previously, `data_generator.start()` would take 30+ seconds for large buffers (>50GB) because Python was trying to pickle and transfer massive tensor objects between processes.

**Solution**: Custom pickling using named shared memory segments:

```python
def __getstate__(self):
    """Only pickle metadata, not the large tensors."""
    state = self.__dict__.copy()
    del state['buffer_tensor']  # Don't pickle 50GB+ tensor
    del state['shm']           # Don't pickle buffer
    return state  # Only send small metadata

def __setstate__(self, state):
    """Reconnect to existing shared memory in child process."""
    self.__dict__.update(state)
    # Connect to SAME memory block using stored name
    self.shm = shared_memory.SharedMemory(name=self.shm_name)
    # Recreate tensor view of identical physical memory
    self.buffer_tensor = torch.frombuffer(self.shm.buf, ...).view(self.shape)
```

**Result**: Process startup now takes milliseconds instead of minutes, enabling practical use with large buffers.

### Shared Memory Architecture

- **Parent Process**: Creates named shared memory segment (`psm_abc123def`)
- **Child Process**: Connects to existing segment by name
- **Memory Access**: Both processes access identical physical RAM
- **Synchronization**: Changes in one process immediately visible in other
- **Cleanup**: Automatic cleanup when all processes release references