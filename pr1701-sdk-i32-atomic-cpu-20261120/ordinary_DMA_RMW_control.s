	.text
	.file	"ordinary_DMA_RMW_control.ll"
	.section	.text.ordinary_DMA_RMW_control,"ax",@progbits
	.globl	ordinary_DMA_RMW_control        # -- Begin function ordinary_DMA_RMW_control
	.type	ordinary_DMA_RMW_control,@function
ordinary_DMA_RMW_control:               # @ordinary_DMA_RMW_control
	.cfi_startproc
# %bb.0:
	addli sp, sp, -128
	addli r3, zero, 4
	lui r4, 983040
	load.d r[1:2], 64(sp)
	addli r5, sp, 0
	load.d r[6:7], 72(sp)
	lui r9, 983040
	or r4, r3, r4
	l_gm2lm r4, r5, 0(r[1:2])
	mfence { lm }
	loadli.w r4,  0(sp)
	addli r8, r4, 1
	storeli.w r8, 0(sp)
	or r8, r3, r9
	mfence { lm }
	s_lm2gm r8, r5, 0(r[1:2])
	lui r1, 983040
	mfence { lm }
	storeli.w r4, 0(sp)
	or r1, r3, r1
	mfence { lm }
	s_lm2gm r1, r5, 0(r[6:7])
	mfence { lm }
	addli sp, sp, 64
	mfence { lm, sm, gm }
	exit
.Lfunc_end0:
	.size	ordinary_DMA_RMW_control, .Lfunc_end0-ordinary_DMA_RMW_control
	.cfi_endproc
                                        # -- End function
	.type	XPU_KERNEL_PARAM_SIZE_ordinary_DMA_RMW_control,@object # @XPU_KERNEL_PARAM_SIZE_ordinary_DMA_RMW_control
	.section	.XPU.KERNEL_PARAM_SIZE.ordinary_DMA_RMW_control,"a",@progbits
	.globl	XPU_KERNEL_PARAM_SIZE_ordinary_DMA_RMW_control
	.p2align	2, 0x0
XPU_KERNEL_PARAM_SIZE_ordinary_DMA_RMW_control:
	.long	16                              # 0x10
	.size	XPU_KERNEL_PARAM_SIZE_ordinary_DMA_RMW_control, 4

	.type	XPU_KERNEL_STACK_SIZE_ordinary_DMA_RMW_control,@object # @XPU_KERNEL_STACK_SIZE_ordinary_DMA_RMW_control
	.section	.XPU.KERNEL_STACK_SIZE.ordinary_DMA_RMW_control,"a",@progbits
	.globl	XPU_KERNEL_STACK_SIZE_ordinary_DMA_RMW_control
	.p2align	2, 0x0
XPU_KERNEL_STACK_SIZE_ordinary_DMA_RMW_control:
	.long	128                             # 0x80
	.size	XPU_KERNEL_STACK_SIZE_ordinary_DMA_RMW_control, 4

	.type	XPU_KERNEL_DYNAMIC_LM_ordinary_DMA_RMW_control,@object # @XPU_KERNEL_DYNAMIC_LM_ordinary_DMA_RMW_control
	.section	.XPU.KERNEL_DYNAMIC_LM.ordinary_DMA_RMW_control,"a",@progbits
	.globl	XPU_KERNEL_DYNAMIC_LM_ordinary_DMA_RMW_control
	.p2align	2, 0x0
XPU_KERNEL_DYNAMIC_LM_ordinary_DMA_RMW_control:
	.long	0                               # 0x0
	.size	XPU_KERNEL_DYNAMIC_LM_ordinary_DMA_RMW_control, 4

	.section	".note.GNU-stack","",@progbits
