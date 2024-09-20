import torch 
import flashinfer
import time 
 

num_layers = 64
num_qo_heads = 64
num_kv_heads = 8
head_dim = 128
page_size = 16
# device="cuda:0"
device="cpu"
# allocate 128MB workspace buffer
workspace_buffer = torch.empty(128 * 1024 * 1024 * 100, dtype=torch.uint8, device=device)
wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
    2, workspace_buffer, "NHD"
)
batch_size = 7
shared_kv_num_pages = 512
unique_kv_num_pages = 128
total_num_pages = shared_kv_num_pages + unique_kv_num_pages
shared_kv_page_indices = torch.arange(shared_kv_num_pages).int().to(device)
shared_kv_page_indptr = torch.tensor([0, shared_kv_num_pages], dtype=torch.int32, device=device)
unique_kv_page_indices = torch.arange(shared_kv_num_pages, total_num_pages).int().to(device)
 
unique_kv_page_indptr = torch.tensor(
    [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device=device
)
shared_kv_last_page_len = torch.tensor([page_size], dtype=torch.int32, device=device)
# 1 <= kv_last_page_len <= page_size
unique_kv_last_page_len = torch.tensor(
    [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device=device
)
kv_cache_at_layer = [
    torch.randn(
        total_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    ) for _ in range(num_layers)
]
qo_indptr_arr = [
    torch.tensor([0, batch_size], dtype=torch.int32, device=device),  # top-level for shared KV-Cache
    torch.arange(batch_size + 1, dtype=torch.int32, device=device)    # bottom-level for unique KV-Cache
]
# create auxiliary data structures for batch decode attention
wrapper.plan(
    qo_indptr_arr,
    [shared_kv_page_indptr, unique_kv_page_indptr],
    [shared_kv_page_indices, unique_kv_page_indices],
    [shared_kv_last_page_len, unique_kv_last_page_len],
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
)
outputs = []

start_time = time.perf_counter()
for _ in range(100):
    for i in range(num_layers):
        print(i)
        q = torch.randn(batch_size, num_qo_heads, head_dim).half().to(device)
        # compute batch decode attention, reuse auxiliary data structures for all layers
        o = wrapper.run(q, kv_cache_at_layer[i])
        outputs.append(o)
end_time = time.perf_counter()
print(f"time elapse: {end_time-start_time}")