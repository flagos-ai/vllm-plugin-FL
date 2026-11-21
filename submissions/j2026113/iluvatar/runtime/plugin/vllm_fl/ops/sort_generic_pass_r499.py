"""Keep native stable eight-pass radix sorting; reuse one generic sweep JIT."""
import hashlib
import inspect
from pathlib import Path


def install_sort_generic_pass():
    import flag_gems
    import triton
    registered=flag_gems.sort
    source=Path(inspect.getfile(registered))
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_hash=='4f63aadbe42f7dbfab0f0fb295c9dea09d749e292ce1b1e61340f68978e8fab9'
    assert registered.__module__=='flag_gems.ops.sort'
    namespace=registered.__globals__
    radix=namespace['radix_sort']
    assert radix.__globals__ is namespace
    original=namespace['sweep']
    assert isinstance(original,triton.runtime.JITFunction)
    assert not getattr(original,'_generic_pass_r499',False)
    assert not original.do_not_specialize
    replacement=triton.jit(do_not_specialize=['pass_id','bit_offset'])(original.fn)
    assert replacement.src==original.src
    assert replacement.arg_names==original.arg_names
    assert [p.is_constexpr for p in replacement.params]==[p.is_constexpr for p in original.params]
    assert replacement.arg_names.index('pass_id')==7 and replacement.arg_names.index('bit_offset')==8
    namespace['sweep']=replacement
    replacement._generic_pass_r499=True
    receipt={'native_registered_module':registered.__module__,'native_source_file':str(source),
             'native_source_sha256':source_hash,'sweep_body_sha256':hashlib.sha256(original.src.encode()).hexdigest(),
             'sweep_source_identical':True,'native_host_loop_histograms_status_and_indices_unchanged':True,
             'do_not_specialize_names':['pass_id','bit_offset'],'do_not_specialize_runtime':list(replacement.do_not_specialize),
             'scope':'Runtime-local replacement of one native JIT instance. Native radix pass count, all kernel bodies, stable order, masks and RNG retained. No installed source, persistent config or class modified.'}
    return {'original':original,'replacement':replacement,'namespace':namespace,'receipt':receipt}
