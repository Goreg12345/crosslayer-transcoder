def all_reduce(gradients, num_gpus, gpu_idx):
    # Split gradients into equal-sized chunks
    chunks = split_into_chunks(gradients, num_gpus)

    # ----- reduce-scatter -----
    for step in range(num_gpus - 1):
        send_idx = (gpu_idx - step) % num_gpus  # chunk we currently own
        recv_idx = (gpu_idx - step - 1) % num_gpus  # chunk arriving this step

        recv_chunk = receive_from_previous_gpu()  # 1. receive
        chunks[recv_idx] += recv_chunk  # 2. reduce into local copy
        send_to_next_gpu(chunks[send_idx])  # 3. send our updated chunk

    # Now each GPU holds the fully reduced chunk whose index == gpu_idx

    # ----- all-gather -----
    for step in range(num_gpus - 1):
        send_idx = (gpu_idx - step + 1) % num_gpus  # chunk we just finished propagating
        recv_idx = (gpu_idx - step) % num_gpus  # chunk about to arrive

        recv_chunk = receive_from_previous_gpu()  # 1. receive new chunk
        chunks[recv_idx] = recv_chunk  # 2. store it
        send_to_next_gpu(chunks[send_idx])  # 3. forward our current chunk

    return combine_chunks(chunks)
