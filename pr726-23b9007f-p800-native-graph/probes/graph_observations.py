import ctypes
import json
import os
from pathlib import Path
import tempfile


def observe_graph(graph, rank, segment):
    runtime = ctypes.CDLL('/usr/local/xpu/so/libcudart.so.12')
    handle = ctypes.c_void_p(graph.raw_cuda_graph())
    count = ctypes.c_size_t()
    runtime.cudaGraphGetNodes.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    runtime.cudaGraphGetNodes.restype = ctypes.c_int
    status = runtime.cudaGraphGetNodes(handle, None, ctypes.byref(count))
    assert status == 0, status
    nodes = (ctypes.c_void_p * count.value)()
    status = runtime.cudaGraphGetNodes(handle, nodes, ctypes.byref(count))
    assert status == 0, status
    runtime.cudaGraphNodeGetType.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    runtime.cudaGraphNodeGetType.restype = ctypes.c_int
    kinds = []
    for node in nodes:
        kind = ctypes.c_int(-1)
        status = runtime.cudaGraphNodeGetType(node, ctypes.byref(kind))
        assert status == 0, status
        kinds.append(kind.value)
    version = ctypes.c_int()
    assert runtime.cudaDriverGetVersion(ctypes.byref(version)) == 0
    observation = {'rank': rank, 'segment': segment, 'node_types': kinds, 'node_count': count.value, 'driver_api_version': version.value,
                   'selected_env': {name: os.environ.get(name) for name in ('BKCL_INFERENCE', 'BKCL_ENABLE_IBGDA', 'BKCL_ENABLE_XDR', 'BKCL_USE_RDMA')}}
    os.write(1, ('GRAPH_OBSERVATION ' + json.dumps(observation, sort_keys=True) + '\n').encode())
    runtime.cudaGraphDebugDotPrint.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
    runtime.cudaGraphDebugDotPrint.restype = ctypes.c_int
    path = Path(tempfile.gettempdir()) / f'flagcx729-graph-{rank}-{segment}.dot'
    status = runtime.cudaGraphDebugDotPrint(handle, str(path).encode(), 1)
    os.write(1, (f'GRAPH_DOT_STATUS {rank} {segment} {status}\n').encode())
    if status == 0 and path.is_file():
        for line in path.read_text().splitlines():
            os.write(1, (f'GRAPH_DOT {rank} {segment} {line}\n').encode())
