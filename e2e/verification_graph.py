"""Fixed M=15/M=16 target verification graphs with logical KV rollback.

Capture cost belongs to the first use (normally benchmark warmup); subsequent
generate calls reuse the same physical cache and graphs. No proposal/acceptance
decision is captured. TD hooks retain graph-owned hidden tensors, restored on
every replay before feature reconstruction.
"""
from collections import Counter
import ctypes
from functools import lru_cache
import time
import torch


@lru_cache(maxsize=1)
def _hip_graph_api():
    """Read-only HIP graph introspection, using the installed runtime ABI."""
    library = ctypes.CDLL("libamdhip64.so")
    library.hipGraphGetNodes.argtypes = [ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    library.hipGraphNodeGetType.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    library.hipGraphChildGraphNodeGetGraph.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    for name in ("hipGraphGetNodes", "hipGraphNodeGetType", "hipGraphChildGraphNodeGetGraph"):
        getattr(library, name).restype = ctypes.c_int
    return library


def graph_node_metadata(graph):
    """Count captured nodes even when ROCTracer omits graph child kernels.

    These are graph-topology counts, not hardware performance counters or
    trace-event counts. Each kernel node executes once per ordinary replay.
    Enum values come from the local hip_runtime_api.h hipGraphNodeType ABI.
    """
    library = _hip_graph_api()
    names = ("kernel", "memcpy", "memset", "host", "child_graph", "empty",
             "wait_event", "event_record", "semaphore_signal", "semaphore_wait",
             "mem_alloc", "mem_free", "memcpy_from_symbol", "memcpy_to_symbol",
             "batch_mem_op")
    histogram = Counter()

    def checked(name, *arguments):
        status = getattr(library, name)(*arguments)
        if status:
            raise RuntimeError(f"{name} failed with HIP error {status}")

    def visit(handle):
        count = ctypes.c_size_t()
        checked("hipGraphGetNodes", handle, None, ctypes.byref(count))
        nodes = (ctypes.c_void_p * count.value)()
        checked("hipGraphGetNodes", handle, nodes, ctypes.byref(count))
        for node in nodes[:count.value]:
            kind = ctypes.c_int()
            checked("hipGraphNodeGetType", node, ctypes.byref(kind))
            histogram[names[kind.value] if kind.value < len(names) else str(kind.value)] += 1
            if kind.value == 4:
                child = ctypes.c_void_p()
                checked("hipGraphChildGraphNodeGetGraph", node, ctypes.byref(child))
                visit(child)

    visit(ctypes.c_void_p(graph.raw_cuda_graph()))
    return {"source": "HIP graph node enumeration, recursively including child graphs",
            "node_types": dict(histogram), "kernel_nodes": histogram["kernel"],
            "total_nodes_including_child_containers": sum(histogram.values())}


class VerificationGraph:
    def __init__(self, target, cache, ids, collector=None):
        if ids.shape[0] != cache.batch_size or ids.shape[1] not in (15, 16):
            raise ValueError("verification graph requires M=15 or M=16")
        if not cache._static_metadata_enabled or any(cache._needs_init):
            raise ValueError("verification graph requires initialized static metadata")
        self.cache, self.collector = cache, collector
        self.ids = torch.empty_like(ids)
        self.rows = ids.shape[1]
        base = cache.length
        self.ids.copy_(ids)
        started = time.perf_counter()

        def forward():
            return target(input_ids=self.ids, past_key_values=cache,
                use_cache=True, attention_mask=None, return_dict=True,
                output_hidden_states=False)

        # Warm all lazy weight layouts/library workspaces before capture.
        for _ in range(2):
            cache.length = base
            forward()
        torch.cuda.synchronize()
        cache.length = base
        cache._metadata_base.fill_(base)
        cache._verification_graph_capture = True
        # Retain the graph for exact HIP node enumeration: ROCTracer can omit
        # child-kernel events on replay and cannot supply a truthful count.
        self.graph = torch.cuda.CUDAGraph(keep_graph=True)
        try:
            with torch.cuda.graph(self.graph):
                self.output = forward()
            self.graph.instantiate()
            self.node_metadata = graph_node_metadata(self.graph)
            self.hidden = dict(collector.values) if collector is not None else None
        finally:
            cache._verification_graph_capture = False
            cache.length = base
        torch.cuda.synchronize()
        self.capture_ms = (time.perf_counter() - started) * 1000

    def replay(self, ids):
        cache = self.cache
        if ids.shape != self.ids.shape or ids.dtype != self.ids.dtype:
            raise ValueError("verification graph input changed shape or dtype")
        if cache.length < 0 or cache.length + self.rows > cache.max_seq_len:
            raise ValueError("verification graph exceeds cache capacity")
        base = cache.length
        self.ids.copy_(ids)
        cache._metadata_base.fill_(base)
        self.graph.replay()
        cache.length = base + self.rows
        cache._active_chunk_metadata = cache._chunk_metadata[self.rows]
        cache._metadata_prepared_key = (base, self.rows)
        if cache._transaction is not None:
            cache._transaction.proposed_length = cache.length
        if self.collector is not None:
            self.collector.values = dict(self.hidden)
        return self.output
