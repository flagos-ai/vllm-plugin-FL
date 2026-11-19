import ctypes
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observe_runtime(rank, comm=None):
    # Observe existing loaded libraries; do not dlopen a second SDK instance.
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and Path(parts[5]).name.startswith(("libbkcl.so", "libflagcx.so")):
            paths.add(parts[5])
    libraries = [
        {"path": path, "sha256": file_hash(path)} for path in sorted(paths)
    ]
    plugin_modules = sorted(name for name in sys.modules if name.startswith("vllm_fl"))
    homo = {"available": False, "reason": "Native PG exposes no public communicator handle"}
    if comm is not None:
        # Exact 6fe header: flagcxResult_t flagcxIsHomoComm(flagcxComm_t, int*).
        # Exact packaged API defines flagcxComm_t = ctypes.c_void_p; comm.comm
        # is the wrapper's own public handle returned by flagcxCommInitRank.
        from flagcx.api import flagcxComm_t

        try:
            query = comm.flagcx.lib["flagcxIsHomoComm"]
            query.argtypes = [flagcxComm_t, ctypes.POINTER(ctypes.c_int)]
            query.restype = ctypes.c_int
            value = ctypes.c_int(-1)
            result = query(comm.comm, ctypes.byref(value))
            homo = {"available": True, "result": result, "value": value.value}
        except Exception as error:
            homo = {"available": False, "error": repr(error)}
    observation = {
        "rank": rank,
        "flagcx_package": version("flagcx"),
        "libraries": libraries,
        "is_homo": homo,
        "plugin_imported": bool(plugin_modules),
        "route": "direct" if comm is not None else "native_pg",
    }
    print("RUNTIME_OBSERVATION", json.dumps(observation, sort_keys=True), flush=True)
    assert observation["flagcx_package"] == "0.14.1+xre5.37.1"
    native = [x for x in libraries if Path(x["path"]).name.startswith("libflagcx.so")]
    assert len(native) == 1 and native[0]["sha256"] == "8618fc37bd45f13f5726b7c5bea04c096828191e27fa95d8f9956c9a23cf3934"
    assert len([x for x in libraries if Path(x["path"]).name.startswith("libbkcl.so")]) == 1
    if comm is None:
        assert not plugin_modules, "Native PG control must not import the plugin"
