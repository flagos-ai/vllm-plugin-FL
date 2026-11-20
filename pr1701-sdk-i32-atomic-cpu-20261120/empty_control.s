	.text
	.file	"empty_control.ll"
	.section	.text.empty_control,"ax",@progbits
	.globl	empty_control                   # -- Begin function empty_control
	.type	empty_control,@function
empty_control:                          # @empty_control
	.cfi_startproc
# %bb.0:
	addli sp, sp, -64
	mfence { lm, sm, gm }
	exit
.Lfunc_end0:
	.size	empty_control, .Lfunc_end0-empty_control
	.cfi_endproc
                                        # -- End function
	.type	XPU_KERNEL_PARAM_SIZE_empty_control,@object # @XPU_KERNEL_PARAM_SIZE_empty_control
	.section	.XPU.KERNEL_PARAM_SIZE.empty_control,"a",@progbits
	.globl	XPU_KERNEL_PARAM_SIZE_empty_control
	.p2align	2, 0x0
XPU_KERNEL_PARAM_SIZE_empty_control:
	.long	16                              # 0x10
	.size	XPU_KERNEL_PARAM_SIZE_empty_control, 4

	.type	XPU_KERNEL_STACK_SIZE_empty_control,@object # @XPU_KERNEL_STACK_SIZE_empty_control
	.section	.XPU.KERNEL_STACK_SIZE.empty_control,"a",@progbits
	.globl	XPU_KERNEL_STACK_SIZE_empty_control
	.p2align	2, 0x0
XPU_KERNEL_STACK_SIZE_empty_control:
	.long	64                              # 0x40
	.size	XPU_KERNEL_STACK_SIZE_empty_control, 4

	.type	XPU_KERNEL_DYNAMIC_LM_empty_control,@object # @XPU_KERNEL_DYNAMIC_LM_empty_control
	.section	.XPU.KERNEL_DYNAMIC_LM.empty_control,"a",@progbits
	.globl	XPU_KERNEL_DYNAMIC_LM_empty_control
	.p2align	2, 0x0
XPU_KERNEL_DYNAMIC_LM_empty_control:
	.long	0                               # 0x0
	.size	XPU_KERNEL_DYNAMIC_LM_empty_control, 4

	.section	".note.GNU-stack","",@progbits
