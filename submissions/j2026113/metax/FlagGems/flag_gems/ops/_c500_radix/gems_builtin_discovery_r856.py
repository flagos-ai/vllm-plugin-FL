"""Task-local JIT dependency discovery for explicitly marked project builtins.

Pinned MACA runtime recognises builtins by module-name prefix in visit_Call,
while its core.builtin decorator already sets __triton_builtin__. Respect
that existing marker in a private finder and per-project JIT subclass only.
No installed runtime, global finder or public Gems registration is modified.
"""
import hashlib
import inspect
from pathlib import Path

import triton.runtime.jit as runtime


def install_project_builtin_discovery(namespace):
    source=inspect.getsource(runtime.DependenciesFinder)
    anchor='return module.startswith(TRITON_MODULE)'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+' or getattr(func, "__triton_builtin__", False)')
    private=dict(runtime.__dict__)
    exec(compile(source,str(Path(__file__))+'::private_dependencies','exec'),private)
    Finder=private['DependenciesFinder']
    root=Path(__file__).resolve().parent

    class ProjectJITFunction(runtime.JITFunction):
        @property
        def cache_key(self):
            if self.hash is None:
                finder=Finder(name=self.__name__,globals=self.__globals__,src=self.src)
                finder.visit(self.parse())
                self.hash=finder.ret+str(self.starting_line_number)
                self.used_global_vals=dict(sorted(finder.used_global_vals.items()))
            return self.hash

    todo=[v for v in namespace.values() if isinstance(v,runtime.JITFunction)]
    seen=set()
    while todo:
        function=todo.pop()
        if id(function) in seen:continue
        seen.add(id(function))
        if Path(inspect.getsourcefile(function.fn)).resolve().parent!=root:continue
        assert function.hash is None,'Install before any compilation or dependency traversal'
        function.__class__=ProjectJITFunction
        todo.extend(v for v in function.fn.__globals__.values() if isinstance(v,runtime.JITFunction))
    assert seen
    return len(seen)
