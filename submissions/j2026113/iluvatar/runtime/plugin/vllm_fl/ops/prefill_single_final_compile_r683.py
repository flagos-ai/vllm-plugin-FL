"""Compose Q reuse and paged V IR edits before a single final codegen."""
import hashlib
import json
from pathlib import Path

try:
    from .prefill_paged_V_SME_K_lifetime_r671 import rewrite,delay_long_K_staging
except ImportError:
    from prefill_paged_V_SME_K_lifetime_r671 import rewrite,delay_long_K_staging


def transform(text,target,directory):
    from triton._C.libtriton import ir,passes
    from triton.backends import backends
    assert target.backend=='corex' and target.arch==71 and target.warp_size==64
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    changed,receipt=rewrite(text)
    digest=hashlib.sha256(changed.encode()).hexdigest();p=directory/(digest+'.ttgir');p.write_text(changed)
    context=ir.context();ir.load_dialects(context);backends['iluvatar'].compiler(target).load_dialects(context)
    module=ir.parse_mlir_module(str(p),context);module.context=context
    pm=ir.pass_manager(context);passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.common.add_canonicalizer(pm);passes.common.add_cse(pm)
    pm.run(module,'Combined_Q_reuse_and_paged_V_transform')
    changed,schedule=delay_long_K_staging(str(module));p.write_text(changed)
    from .prefill_native_K_maskfirst_r970 import rewrite as rewrite_K_pages
    changed,K_receipt=rewrite_K_pages(changed)
    receipt['native_paged_K']=K_receipt
    p.write_text(changed)
    assert not any(' = tt.load ' in line and 'tensor<16x128x!tt.ptr<bf16>' in line for line in changed.splitlines())
    receipt.update(K_staging_schedule=schedule,composition='IR edits composed before existing final compile; no intermediate Q-only binary generated',
        input_ir_SHA256=hashlib.sha256(text.encode()).hexdigest(),output_ir_SHA256=hashlib.sha256(changed.encode()).hexdigest(),ir_path=str(p))
    (directory/(digest+'.json')).write_text(json.dumps(receipt,indent=2))
    return changed,receipt


def patch_compiler_source(patched_source):
    anchor="        digest = hashlib.sha256((RULE_VERSION+'\\n'+changed_ir).encode()).hexdigest()"
    assert patched_source.count(anchor)==1
    addition="""        if prefill and constants.get('BLOCK_SIZE')==16 and constants.get('KV_QUANT_MODE')==0:
            from prefill_single_final_compile_r680 import transform as _paged_compose_transform
            changed_ir,_paged_compose_receipt = _paged_compose_transform(changed_ir,target,
                Path(os.environ['TIANSHU_Q_OPERAND_CACHE'])/'composed_prefill_V')
            receipt['composed_paged_V'] = _paged_compose_receipt
"""
    return patched_source.replace(anchor,addition+anchor)
