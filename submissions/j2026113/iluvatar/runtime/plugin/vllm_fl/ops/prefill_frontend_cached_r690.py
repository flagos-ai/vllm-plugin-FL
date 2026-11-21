"""Generate native TTGIR without an unused original binary, kernel scope only."""
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace


def install(module,root):
    kernel=module.kernel_unified_attention;wrapper=kernel.compile
    cells=dict(zip(wrapper.__code__.co_freevars,wrapper.__closure__))
    assert 'original_compile' in cells and 'previous' not in cells
    original=cells['original_compile'].cell_contents
    root=Path(root);root.mkdir(parents=True,exist_ok=True);records=[]

    def compile_local(src,target=None,options=None,**kwargs):
        constants={src.fn.arg_names[key[0]]:value for key,value in src.constants.items() if len(key)==1}
        eligible=(target.backend=='corex' and target.arch==71 and target.warp_size==64
            and not constants.get('DIRECT_DECODE_SEQ') and not constants.get('IS_3D')
            and constants.get('BLOCK_M')==128 and constants.get('TILE_SIZE')==64
            and constants.get('HEAD_SIZE')==128 and constants.get('BLOCK_SIZE')==16
            and constants.get('KV_QUANT_MODE')==0 and src.signature.get('query_ptr')=='*bf16')
        if not eligible:return original(src,target=target,options=options,**kwargs)
        begin=time.perf_counter()
        if (options or {}).get('ir_override'):
            compiled=original(src,target=target,options=options,**kwargs)
            row={'phase':'final_native_codegen','seconds':time.perf_counter()-begin,'source_hash':src.hash(),
                'kernel_hash':compiled.hash,'shared_bytes':compiled.metadata.shared,
                'final_TTGIR_SHA256':hashlib.sha256(compiled.asm['ttgir'].encode()).hexdigest()}
            records.append(row);(root/'records.json').write_text(json.dumps(records,indent=2));return compiled
        from triton.compiler.compiler import make_backend,get_cache_invalidating_env_vars,get_cache_key,_flagprism
        from triton._C.libtriton import ir
        from triton import __version__
        backend=make_backend(target)
        raw_options=dict(options or {},**src.parse_options());_flagprism.apply_compile_options(raw_options)
        parsed=backend.parse_options(raw_options)
        assert parsed.num_warps==8 and parsed.num_stages==1
        env_vars=kwargs.get('_env_vars') or get_cache_invalidating_env_vars()
        native_key=get_cache_key(src,backend,parsed,env_vars)
        key=hashlib.sha256(('native_prefill_frontend_cache_v1\n'+native_key).encode()).hexdigest()
        cache_metadata=root/(key+'.cache.json');cache_ir=root/(key+'.native.ttgir')
        if cache_metadata.exists():
            saved=json.loads(cache_metadata.read_text());text=cache_ir.read_text()
            assert saved['native_key_SHA256']==hashlib.sha256(native_key.encode()).hexdigest()
            assert saved['TTGIR_SHA256']==hashlib.sha256(text.encode()).hexdigest()
            row={'phase':'native_frontend_only','seconds':time.perf_counter()-begin,'source_hash':src.hash(),
                'frontend_cache_hit':True,'TTGIR_SHA256':saved['TTGIR_SHA256'],'ir_path':str(cache_ir),
                'original_LLIR_ASM_binary_not_generated':True,'shared_memory_before_final_codegen':None,
                'cache_key_includes':'Native triton_key/source/target/backend_options/env invalidation key'}
            records.append(row);(root/'records.json').write_text(json.dumps(records,indent=2))
            return SimpleNamespace(asm={'ttgir':text},metadata=SimpleNamespace(num_warps=parsed.num_warps,num_stages=parsed.num_stages,shared=None))
        metadata={'target':target,**parsed.__dict__,**env_vars,'triton_version':__version__}
        stages={};backend.add_stages(stages,parsed,src.language)
        assert list(stages)[:2]==['ttir','ttgir'],list(stages)
        context=ir.context();ir.load_dialects(context);backend.load_dialects(context);_flagprism.load_dialects(context)
        codegen=backend.get_codegen_implementation(parsed);module_map=backend.get_module_map()
        native=src.make_ir(target,parsed,codegen,module_map,context)
        phase_times={}
        for ext in ['ttir','ttgir']:
            started=time.perf_counter();native=stages[ext](native,metadata);phase_times[ext]=time.perf_counter()-started
        text=str(native);digest=hashlib.sha256(text.encode()).hexdigest();path=cache_ir
        temp_ir=root/(key+f'.{os.getpid()}.ir.tmp');temp_ir.write_text(text);os.replace(temp_ir,cache_ir)
        saved={'native_key_SHA256':hashlib.sha256(native_key.encode()).hexdigest(),'TTGIR_SHA256':digest}
        temp_metadata=root/(key+f'.{os.getpid()}.json.tmp');temp_metadata.write_text(json.dumps(saved,indent=2));os.replace(temp_metadata,cache_metadata)
        row={'phase':'native_frontend_only','seconds':time.perf_counter()-begin,'stage_seconds':phase_times,'frontend_cache_hit':False,
            'source_hash':src.hash(),'TTGIR_SHA256':digest,'ir_path':str(path),
            'stages_executed':['make_ir','ttir','ttgir'],'original_LLIR_ASM_binary_not_generated':True,
            'shared_memory_before_final_codegen':None}
        records.append(row);(root/'records.json').write_text(json.dumps(records,indent=2))
        # The existing local IR editor needs only TTGIR + warp/stage metadata.
        # Shared-memory usage is unavailable until final lowering; leave null.
        return SimpleNamespace(asm={'ttgir':text},metadata=SimpleNamespace(num_warps=parsed.num_warps,
            num_stages=parsed.num_stages,shared=None))

    cells['original_compile'].cell_contents=compile_local
    return records
