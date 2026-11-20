	.text
	.file	"LLVMDialectModule"
	.section	.text._bincount_kernel,"ax",@progbits
	.globl	_bincount_kernel                # -- Begin function _bincount_kernel
	.type	_bincount_kernel,@function
_bincount_kernel:                       # @_bincount_kernel
.Lfunc_begin0:
	.file	1 "/flagos/lib/python3.10/site-packages/vllm/v1/worker/gpu/sample" "penalties.py"
	.loc	1 217 0                         # penalties.py:217:0
	.cfi_sections .debug_frame
	.cfi_startproc
# %bb.0:
	addli sp, sp, -424
	addli r30, sp, 296
	load_param_i r3, 0
	loadli.w r1,  72(r30)
	loadli.w r2,  68(r30)
	loadli.w r4,  76(r30)
	mul.s r2, r1, r2
	mul.s r4, r2, r4
	ble.s r4, r3, .LBB0_17
# %bb.1:                                # %.lr.ph
	loadli.w r6,  16(r30)
	loadli.w r2,  64(r30)
	loadli.w r5,  48(r30)
	load_param_i r1, 1
	addli r27, zero, 2
	addli r28, r30, -192
	addli r8, zero, 32
	addli r17, r30, -64
	addli r13, zero, 1024
	storeli.w r1, -268(r30)                 # 4-byte Folded Spill
	sext_int2long r[6:7], r6
	sext_int2long r[1:2], r2
	store.d r[6:7], -280(r30)               # 8-byte Folded Spill
	sext_int2long r[5:6], r5
	store.d r[1:2], -296(r30)               # 8-byte Folded Spill
	addli r7, r30, -128
	addli r1, zero, 1
	store.d r[5:6], -288(r30)               # 8-byte Folded Spill
	addli r5, zero, 4
	core_id r6
	j .LBB0_3
.LBB0_2:                                # %.loopexit
                                        #   in Loop: Header=BB0_3 Depth=1
	loadli.w r2,  -268(r30)                 # 4-byte Folded Reload
	add.s r3, r3, r2
	blt.s r3, r4, .LBB0_3
	j .LBB0_17
.LBB0_3:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_8 Depth 2
	loadli.w r2,  76(r30)
	loadli.w r9,  72(r30)
.Ltmp0:
	.loc	1 231 55 prologue_end           # penalties.py:231:55
	addli r12, zero, 3
	load.d r[15:16], 0(r30)
	.loc	1 231 28 is_stmt 0              # penalties.py:231:28
	lui r14, 983040
	div.s r2, r3, r2
	addli r11, r9, 0
	div.s r9, r2, r11
	mod.s r2, r2, r11
	.loc	1 231 55                        # penalties.py:231:55
	sext_int2long r[9:10], r9
	sll.d r[9:10], r[9:10], r12
	.loc	1 231 28                        # penalties.py:231:28
	addli r12, zero, 8
	or r14, r12, r14
	.loc	1 231 55                        # penalties.py:231:55
	add.ul r[9:10], r[15:16], r[9:10]
	.loc	1 231 28                        # penalties.py:231:28
	addli r12, r30, -256
	l_gm2lm r14, r12, 0(r[9:10])
	.loc	1 233 26 is_stmt 1              # penalties.py:233:26
	lui r9, 983040
	.loc	1 231 28                        # penalties.py:231:28
	mfence { lm }
	.loc	1 233 44                        # penalties.py:233:44
	load.d r[10:11], 32(r30)
	.loc	1 231 28                        # penalties.py:231:28
	load.d r[18:19], -256(r30)
	.loc	1 233 26                        # penalties.py:233:26
	or r9, r5, r9
	.loc	1 233 44 is_stmt 0              # penalties.py:233:44
	sll.d r[15:16], r[18:19], r27
	add.ul r[20:21], r[10:11], r[15:16]
	.loc	1 233 26                        # penalties.py:233:26
	l_gm2lm r9, r28, 0(r[20:21])
	.loc	1 234 19 is_stmt 1              # penalties.py:234:19
	slli r9, r2, 10
	.loc	1 233 26                        # penalties.py:233:26
	mfence { lm }
	loadli.w r14,  -192(r30)
	mfence { lm }
	.loc	1 234 33                        # penalties.py:234:33
	ble.s r14, r9, .LBB0_2
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 33 is_stmt 0                # penalties.py:0:33
	load.d r[10:11], -280(r30)              # 8-byte Folded Reload
	addli r2, zero, 0
	mul.sl r[20:21], r[18:19], r[10:11]
	.loc	1 237 42 is_stmt 1              # penalties.py:237:42
	load.d r[10:11], 24(r30)
	sll.d r[20:21], r[20:21], r27
	add.ul r[15:16], r[10:11], r[15:16]
	load.d r[11:12], 8(r30)
	.loc	1 253 25                        # penalties.py:253:25
	addli r10, r9, 1024
	add.ul r[24:25], r[11:12], r[20:21]
	load.d r[11:12], -288(r30)              # 8-byte Folded Reload
	mul.sl r[20:21], r[18:19], r[11:12]
	load.d r[11:12], -296(r30)              # 8-byte Folded Reload
	sll.d r[20:21], r[20:21], r27
	mul.sl r[18:19], r[18:19], r[11:12]
	load.d r[11:12], 40(r30)
	sll.d r[22:23], r[18:19], r27
	add.ul r[18:19], r[11:12], r[20:21]
	load.d r[11:12], 56(r30)
	add.ul r[11:12], r[11:12], r[22:23]
	store.d r[11:12], -264(r30)             # 8-byte Folded Spill
	j .LBB0_8
.LBB0_5:                                # %.critedge.critedge
                                        #   in Loop: Header=BB0_8 Depth=2
	.loc	1 257 12                        # penalties.py:257:12
	mfence { lm }
	mfence { lm }
	.loc	1 263 12                        # penalties.py:263:12
	mfence { lm }
	loadli.w r11,  -128(r30)
	mfence { lm }
	addli r11, r11, 1
	storeli.w r11, -128(r30)
	mfence { lm }
.LBB0_6:                                #   in Loop: Header=BB0_8 Depth=2
	mfence { lm }
.LBB0_7:                                #   in Loop: Header=BB0_8 Depth=2
	.loc	1 0 12 is_stmt 0                # penalties.py:0:12
	addli r2, r2, 1
	beq r2, r13, .LBB0_2
.LBB0_8:                                #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 237 25 is_stmt 1              # penalties.py:237:25
	lui r22, 983040
	.loc	1 239 32                        # penalties.py:239:32
	add.s r26, r9, r2
	.loc	1 237 25                        # penalties.py:237:25
	or r22, r5, r22
	l_gm2lm r22, r28, 0(r[15:16])
	mfence { lm }
	loadli.w r22,  -192(r30)
	mfence { lm }
	.loc	1 239 7                         # penalties.py:239:7
	ble.s r22, r26, .LBB0_12
# %bb.9:                                #   in Loop: Header=BB0_8 Depth=2
	ble.s r22, r9, .LBB0_12
# %bb.10:                               #   in Loop: Header=BB0_8 Depth=2
	bneq r6, zero, .LBB0_12
# %bb.11:                               #   in Loop: Header=BB0_8 Depth=2
	.loc	1 242 71                        # penalties.py:242:71
	sext_int2long r[11:12], r26
	.loc	1 242 12 is_stmt 0              # penalties.py:242:12
	lui r23, 983040
	.loc	1 242 71                        # penalties.py:242:71
	sll.d r[11:12], r[11:12], r27
	.loc	1 242 12                        # penalties.py:242:12
	or r23, r5, r23
	.loc	1 242 71                        # penalties.py:242:71
	add.ul r[11:12], r[24:25], r[11:12]
	.loc	1 242 12                        # penalties.py:242:12
	l_gm2lm r23, r7, 0(r[11:12])
	.loc	1 249 12 is_stmt 1              # penalties.py:249:12
	lui r23, 983040
	.loc	1 242 12                        # penalties.py:242:12
	mfence { lm }
	loadli.w r11,  -128(r30)
	mfence { lm }
	.loc	1 244 31                        # penalties.py:244:31
	div.s r12, r11, r8
	.loc	1 246 53                        # penalties.py:246:53
	andi r11, r11, 31
	sll r11, r1, r11
	.loc	1 248 75                        # penalties.py:248:75
	sext_int2long r[20:21], r12
	.loc	1 249 12                        # penalties.py:249:12
	or r12, r5, r23
	.loc	1 248 75                        # penalties.py:248:75
	sll.d r[20:21], r[20:21], r27
	add.ul r[20:21], r[18:19], r[20:21]
	.loc	1 249 12                        # penalties.py:249:12
	l_gm2lm r12, r17, 0(r[20:21])
	mfence { lm }
	loadli.w r12,  -64(r30)
	or r11, r12, r11
	lui r12, 983040
	storeli.w r11, -128(r30)
	or r11, r5, r12
	mfence { lm }
	s_lm2gm r11, r7, 0(r[20:21])
	mfence { lm }
.LBB0_12:                               #   in Loop: Header=BB0_8 Depth=2
	.loc	1 253 7                         # penalties.py:253:7
	blt.s r10, r22, .LBB0_7
# %bb.13:                               #   in Loop: Header=BB0_8 Depth=2
	bneq r6, zero, .LBB0_7
# %bb.14:                               #   in Loop: Header=BB0_8 Depth=2
	.loc	1 257 12                        # penalties.py:257:12
	ble.s r14, r26, .LBB0_5
# %bb.15:                               #   in Loop: Header=BB0_8 Depth=2
	blt.s r26, r22, .LBB0_5
# %bb.16:                               #   in Loop: Header=BB0_8 Depth=2
	.loc	1 257 71 is_stmt 0              # penalties.py:257:71
	sext_int2long r[11:12], r26
	.loc	1 257 12                        # penalties.py:257:12
	lui r20, 983040
	.loc	1 257 71                        # penalties.py:257:71
	sll.d r[11:12], r[11:12], r27
	.loc	1 257 12                        # penalties.py:257:12
	or r20, r5, r20
	.loc	1 257 71                        # penalties.py:257:71
	add.ul r[11:12], r[24:25], r[11:12]
	.loc	1 257 12                        # penalties.py:257:12
	l_gm2lm r20, r7, 0(r[11:12])
	.loc	1 263 12 is_stmt 1              # penalties.py:263:12
	lui r20, 983040
	.loc	1 257 12                        # penalties.py:257:12
	mfence { lm }
	loadext.w r[11:12], -128(r30)
	.loc	1 262 14                        # penalties.py:262:14
	load.d r[21:22], -264(r30)              # 8-byte Folded Reload
	.loc	1 263 12                        # penalties.py:263:12
	or r20, r5, r20
	.loc	1 257 12                        # penalties.py:257:12
	mfence { lm }
	.loc	1 262 14                        # penalties.py:262:14
	sll.d r[11:12], r[11:12], r27
	add.ul r[11:12], r[21:22], r[11:12]
	.loc	1 263 12                        # penalties.py:263:12
	lui r21, 983040
	l_gm2lm r20, r7, 0(r[11:12])
	mfence { lm }
	loadli.w r20,  -128(r30)
	mfence { lm }
	addli r20, r20, 1
	storeli.w r20, -128(r30)
	or r20, r5, r21
	mfence { lm }
	s_lm2gm r20, r7, 0(r[11:12])
	j .LBB0_6
.LBB0_17:                               # %._crit_edge
	.loc	1 235 8 epilogue_begin          # penalties.py:235:8
	addli sp, sp, 296
	mfence { lm, sm, gm }
	exit
.Ltmp1:
.Lfunc_end0:
	.size	_bincount_kernel, .Lfunc_end0-_bincount_kernel
	.cfi_endproc
                                        # -- End function
	.type	XPU_KERNEL_PARAM_SIZE__bincount_kernel,@object # @XPU_KERNEL_PARAM_SIZE__bincount_kernel
	.section	.XPU.KERNEL_PARAM_SIZE._bincount_kernel,"a",@progbits
	.globl	XPU_KERNEL_PARAM_SIZE__bincount_kernel
	.p2align	2, 0x0
XPU_KERNEL_PARAM_SIZE__bincount_kernel:
	.long	80                              # 0x50
	.size	XPU_KERNEL_PARAM_SIZE__bincount_kernel, 4

	.type	XPU_KERNEL_STACK_SIZE__bincount_kernel,@object # @XPU_KERNEL_STACK_SIZE__bincount_kernel
	.section	.XPU.KERNEL_STACK_SIZE._bincount_kernel,"a",@progbits
	.globl	XPU_KERNEL_STACK_SIZE__bincount_kernel
	.p2align	2, 0x0
XPU_KERNEL_STACK_SIZE__bincount_kernel:
	.long	424                             # 0x1a8
	.size	XPU_KERNEL_STACK_SIZE__bincount_kernel, 4

	.type	XPU_KERNEL_DYNAMIC_LM__bincount_kernel,@object # @XPU_KERNEL_DYNAMIC_LM__bincount_kernel
	.section	.XPU.KERNEL_DYNAMIC_LM._bincount_kernel,"a",@progbits
	.globl	XPU_KERNEL_DYNAMIC_LM__bincount_kernel
	.p2align	2, 0x0
XPU_KERNEL_DYNAMIC_LM__bincount_kernel:
	.long	0                               # 0x0
	.size	XPU_KERNEL_DYNAMIC_LM__bincount_kernel, 4

	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	37                              # DW_AT_producer
	.byte	14                              # DW_FORM_strp
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	4                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x1b DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	2                               # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.long	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
.asciz	"triton"                        # string offset=0
.Linfo_string1:
.asciz	"penalties.py"                  # string offset=7
.Linfo_string2:
.asciz	"/flagos/lib/python3.10/site-packages/vllm/v1/worker/gpu/sample" # string offset=20
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
