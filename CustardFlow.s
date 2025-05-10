	.file	"CustardFlow.c"
	.text
	.p2align 4
	.globl	naive_matmul
	.type	naive_matmul, @function
naive_matmul:
.LFB5307:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	72(%rsp), %r14
	testq	%rcx, %rcx
	je	.L16
	movq	56(%rsp), %rax
	movq	%rsi, %rbx
	leaq	0(,%rax,4), %r13
	movq	64(%rsp), %rax
	movq	%r9, %rsi
	movq	%rdx, %rbp
	movq	%rcx, %r12
	movq	%r8, %r10
	leaq	0(,%rax,4), %r9
	xorl	%r15d, %r15d
	xorl	%r11d, %r11d
	.p2align 4,,10
	.p2align 3
.L3:
	testq	%r10, %r10
	je	.L5
	movq	%rbx, %rcx
	leaq	0(%rbp,%r15,4), %rdx
	xorl	%r8d, %r8d
	.p2align 4,,10
	.p2align 3
.L8:
	testq	%rsi, %rsi
	je	.L7
	vmovss	(%rdx), %xmm0
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L4:
	vmovss	(%rdi,%rax,4), %xmm1
	vfmadd231ss	(%rcx,%rax,4), %xmm1, %xmm0
	incq	%rax
	vmovss	%xmm0, (%rdx)
	cmpq	%rax, %rsi
	jne	.L4
.L7:
	incq	%r8
	addq	%r9, %rcx
	addq	$4, %rdx
	cmpq	%r8, %r10
	jne	.L8
.L5:
	incq	%r11
	addq	%r13, %rdi
	addq	%r14, %r15
	cmpq	%r11, %r12
	jne	.L3
.L16:
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE5307:
	.size	naive_matmul, .-naive_matmul
	.p2align 4
	.globl	outer_product_matmul
	.type	outer_product_matmul, @function
outer_product_matmul:
.LFB5308:
	.cfi_startproc
	endbr64
	testq	%r9, %r9
	je	.L37
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movq	%r8, %r11
	movq	%rcx, %r10
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	movq	%r9, %r14
	leaq	0(,%r8,4), %r9
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	movq	%rdx, %r13
	movq	%rsi, %r8
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	addq	%r9, %r13
	xorl	%r15d, %r15d
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movq	%r11, %rbp
	negq	%rbp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	salq	$2, %rbp
	movq	%rdi, %rbx
	xorl	%r12d, %r12d
	.p2align 4,,10
	.p2align 3
.L21:
	testq	%r10, %r10
	je	.L23
	leaq	(%rbx,%r15,4), %rsi
	movq	%r13, %rcx
	xorl	%edi, %edi
	.p2align 4,,10
	.p2align 3
.L26:
	leaq	(%rcx,%rbp), %rax
	movq	%r8, %rdx
	testq	%r11, %r11
	je	.L25
	.p2align 4,,10
	.p2align 3
.L22:
	vmovss	(%rax), %xmm1
	vmovss	(%rsi), %xmm0
	addq	$4, %rax
	vfmadd132ss	(%rdx), %xmm1, %xmm0
	addq	$4, %rdx
	vmovss	%xmm0, -4(%rax)
	cmpq	%rcx, %rax
	jne	.L22
.L25:
	incq	%rdi
	addq	$4, %rsi
	addq	%r9, %rcx
	cmpq	%rdi, %r10
	jne	.L26
.L23:
	incq	%r12
	addq	%r9, %r8
	addq	%r10, %r15
	cmpq	%r12, %r14
	jne	.L21
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.L37:
	.cfi_restore 3
	.cfi_restore 6
	.cfi_restore 12
	.cfi_restore 13
	.cfi_restore 14
	.cfi_restore 15
	ret
	.cfi_endproc
.LFE5308:
	.size	outer_product_matmul, .-outer_product_matmul
	.p2align 4
	.globl	tiled_matmul
	.type	tiled_matmul, @function
tiled_matmul:
.LFB5309:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$32, %rsp
	.cfi_def_cfa_offset 88
	movq	%rdi, 8(%rsp)
	movq	%rsi, 16(%rsp)
	movq	%rdx, 24(%rsp)
	movq	%rcx, -56(%rsp)
	movq	%r8, -88(%rsp)
	testq	%rcx, %rcx
	je	.L72
	movq	88(%rsp), %rbx
	leaq	0(,%r9,4), %rax
	imulq	%r8, %rbx
	movq	%rax, -112(%rsp)
	movq	$0, -80(%rsp)
	movq	%rbx, -16(%rsp)
	movq	88(%rsp), %rbx
	movq	$0, -64(%rsp)
	imulq	%r9, %rbx
	movq	$0, -104(%rsp)
	leaq	0(,%r8,4), %r15
	movq	%rbx, -24(%rsp)
	movq	88(%rsp), %rbx
	vxorps	%xmm1, %xmm1, %xmm1
	leaq	0(,%rbx,4), %rdi
	movq	%r8, %rbx
	imulq	%rdi, %rbx
	movq	%rdi, -40(%rsp)
	movq	%r9, %r14
	movq	%rbx, -8(%rsp)
.L42:
	movq	-104(%rsp), %rax
	movq	88(%rsp), %rbx
	movq	%rax, -32(%rsp)
	addq	%rbx, %rax
	cmpq	$0, -88(%rsp)
	movq	%rax, -104(%rsp)
	je	.L51
	movq	8(%rsp), %rax
	movq	-80(%rsp), %rbx
	movq	$0, -72(%rsp)
	leaq	(%rax,%rbx,4), %rax
	movq	%rax, (%rsp)
	movq	$0, -96(%rsp)
	movq	16(%rsp), %r12
.L53:
	movq	-96(%rsp), %rax
	movq	88(%rsp), %rbx
	movq	%rax, -120(%rsp)
	addq	%rbx, %rax
	movq	%rax, -96(%rsp)
	testq	%r14, %r14
	je	.L49
	movq	-56(%rsp), %rbx
	movq	-104(%rsp), %rbp
	movq	24(%rsp), %rdi
	cmpq	%rbp, %rbx
	cmovbe	%rbx, %rbp
	movq	-88(%rsp), %rbx
	cmpq	%rax, %rbx
	cmovbe	%rbx, %rax
	xorl	%r11d, %r11d
	movq	%rax, %rbx
	movq	-64(%rsp), %rax
	addq	%rbx, %rax
	leaq	(%rdi,%rax,4), %rax
	movq	%rax, -48(%rsp)
	movq	-72(%rsp), %rdi
	movq	%rbx, %rax
	negq	%rax
	leaq	(%rdi,%rax,4), %r13
.L52:
	movq	-32(%rsp), %rax
	movq	%r11, %r8
	movq	(%rsp), %rsi
	movq	-48(%rsp), %r9
	addq	88(%rsp), %r11
	movq	%rax, %r10
	cmpq	%rax, %rbp
	jbe	.L46
	.p2align 4,,10
	.p2align 3
.L50:
	cmpq	-120(%rsp), %rbx
	jbe	.L48
	cmpq	%r11, %r14
	movq	%r11, %rcx
	cmovbe	%r14, %rcx
	leaq	(%r9,%r13), %rdi
	movq	%r12, %rdx
	.p2align 4,,10
	.p2align 3
.L45:
	cmpq	%r8, %rcx
	jbe	.L54
	movq	%r8, %rax
	vmovaps	%xmm1, %xmm0
	.p2align 4,,10
	.p2align 3
.L44:
	vmovss	(%rsi,%rax,4), %xmm2
	vfmadd231ss	(%rdx,%rax,4), %xmm2, %xmm0
	incq	%rax
	cmpq	%rcx, %rax
	jne	.L44
.L43:
	vaddss	(%rdi), %xmm0, %xmm0
	addq	$4, %rdi
	addq	%r15, %rdx
	vmovss	%xmm0, -4(%rdi)
	cmpq	%rdi, %r9
	jne	.L45
.L48:
	incq	%r10
	addq	%r15, %r9
	addq	-112(%rsp), %rsi
	cmpq	%rbp, %r10
	jne	.L50
.L46:
	cmpq	%r11, %r14
	ja	.L52
.L49:
	movq	-40(%rsp), %rbx
	addq	-8(%rsp), %r12
	addq	%rbx, -72(%rsp)
	movq	-96(%rsp), %rbx
	cmpq	%rbx, -88(%rsp)
	ja	.L53
.L51:
	movq	-16(%rsp), %rbx
	addq	%rbx, -64(%rsp)
	movq	-24(%rsp), %rbx
	addq	%rbx, -80(%rsp)
	movq	-104(%rsp), %rbx
	cmpq	%rbx, -56(%rsp)
	ja	.L42
.L72:
	addq	$32, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L54:
	.cfi_restore_state
	vmovaps	%xmm1, %xmm0
	jmp	.L43
	.cfi_endproc
.LFE5309:
	.size	tiled_matmul, .-tiled_matmul
	.p2align 4
	.globl	l1_tiled_matmul
	.type	l1_tiled_matmul, @function
l1_tiled_matmul:
.LFB5310:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$192, %rsp
	.cfi_def_cfa_offset 248
	movq	%rdi, 168(%rsp)
	movq	%rsi, 176(%rsp)
	movq	%rdx, 184(%rsp)
	movq	%rcx, -32(%rsp)
	movq	%r8, -48(%rsp)
	testq	%rcx, %rcx
	je	.L119
	movq	248(%rsp), %rbx
	movq	%r8, %rax
	imulq	%r8, %rbx
	salq	$2, %rax
	movq	%rax, -88(%rsp)
	movq	%rbx, 112(%rsp)
	movq	248(%rsp), %rbx
	leaq	0(,%r9,4), %rax
	imulq	%r9, %rbx
	movq	%rax, -96(%rsp)
	movq	$0, -16(%rsp)
	movq	%rbx, -24(%rsp)
	movq	256(%rsp), %rbx
	movq	$0, -8(%rsp)
	imulq	%r8, %rbx
	movq	$0, -80(%rsp)
	movq	%r9, %r13
	movq	%rbx, 144(%rsp)
	movq	256(%rsp), %rbx
	vxorps	%xmm1, %xmm1, %xmm1
	imulq	%r9, %rbx
	movq	%rbx, 8(%rsp)
	movq	256(%rsp), %rbx
	salq	$2, %rbx
	movq	%rbx, 32(%rsp)
.L76:
	movq	-80(%rsp), %rax
	movq	248(%rsp), %rbx
	movq	%rax, 80(%rsp)
	addq	%rbx, %rax
	cmpq	$0, -48(%rsp)
	movq	%rax, -80(%rsp)
	movq	$0, (%rsp)
	movq	$0, -72(%rsp)
	je	.L92
.L94:
	movq	-72(%rsp), %rbx
	movq	248(%rsp), %rdi
	movq	%rbx, 88(%rsp)
	movq	%rbx, %rax
	addq	%rdi, %rbx
	movq	%rbx, -72(%rsp)
	testq	%r13, %r13
	je	.L90
	movq	-32(%rsp), %rdi
	movq	-80(%rsp), %rbx
	movq	$0, -64(%rsp)
	cmpq	%rbx, %rdi
	cmovbe	%rdi, %rbx
	salq	$2, %rax
	movq	%rbx, 56(%rsp)
	movq	%rax, 120(%rsp)
.L93:
	movq	-64(%rsp), %rbx
	movq	248(%rsp), %rdi
	movq	%rbx, 96(%rsp)
	movq	%rbx, %rax
	addq	%rdi, %rbx
	movq	%rbx, -64(%rsp)
	movq	80(%rsp), %rdi
	movq	56(%rsp), %rbx
	cmpq	%rbx, %rdi
	jnb	.L88
	movq	-48(%rsp), %rsi
	movq	-72(%rsp), %rbx
	movq	%rdi, -56(%rsp)
	cmpq	%rbx, %rsi
	cmovbe	%rsi, %rbx
	salq	$2, %rax
	movq	%rbx, 64(%rsp)
	movq	168(%rsp), %rbx
	movq	%r13, %r14
	addq	%rax, %rbx
	addq	176(%rsp), %rax
	movq	%rax, 152(%rsp)
	movq	-16(%rsp), %rax
	movq	%rbx, 136(%rsp)
	movq	%rax, 16(%rsp)
	movq	-8(%rsp), %rax
	movq	%rax, 24(%rsp)
.L91:
	movq	-56(%rsp), %rax
	movq	256(%rsp), %rbx
	movq	%rax, 104(%rsp)
	addq	%rbx, %rax
	movq	%rax, -56(%rsp)
	movq	88(%rsp), %rbx
	movq	64(%rsp), %rax
	cmpq	%rax, %rbx
	jnb	.L86
	movq	-64(%rsp), %rax
	movq	16(%rsp), %rdi
	cmpq	%rax, %r14
	cmovbe	%r14, %rax
	movq	%rbx, -40(%rsp)
	movq	%rax, 72(%rsp)
	movq	136(%rsp), %rax
	leaq	(%rax,%rdi,4), %rax
	movq	%rax, 128(%rsp)
	movq	120(%rsp), %rax
	movq	%rax, 40(%rsp)
	movq	(%rsp), %rax
	movq	%rax, -120(%rsp)
.L89:
	movq	-40(%rsp), %rax
	movq	256(%rsp), %rbx
	movq	%rax, -104(%rsp)
	movq	96(%rsp), %rdi
	addq	%rbx, %rax
	movq	72(%rsp), %rbx
	movq	%rax, -40(%rsp)
	cmpq	%rbx, %rdi
	jnb	.L84
	movq	-32(%rsp), %rsi
	movq	-56(%rsp), %rbx
	movq	%rdi, %r9
	cmpq	%rbx, %rsi
	cmovbe	%rsi, %rbx
	movq	%rdi, %rsi
	movq	%rbx, -112(%rsp)
	movq	-48(%rsp), %rbx
	cmpq	%rax, %rbx
	cmovbe	%rbx, %rax
	movq	184(%rsp), %rbx
	movq	%rax, %r13
	movq	24(%rsp), %rax
	addq	%r13, %rax
	leaq	(%rbx,%rax,4), %rax
	movq	%rax, 160(%rsp)
	movq	40(%rsp), %rbx
	movq	%r13, %rax
	negq	%rax
	leaq	(%rbx,%rax,4), %r15
	movq	128(%rsp), %rax
	movq	152(%rsp), %rbx
	movq	%rax, 48(%rsp)
.L87:
	movq	104(%rsp), %rax
	movq	-112(%rsp), %rdi
	movq	%rsi, %rbp
	movq	48(%rsp), %r11
	movq	160(%rsp), %r10
	addq	256(%rsp), %rsi
	movq	%rax, %r12
	cmpq	%rdi, %rax
	jnb	.L81
	.p2align 4,,10
	.p2align 3
.L85:
	movq	-120(%rsp), %r8
	leaq	(%r15,%r10), %rdi
	cmpq	%r13, -104(%rsp)
	jnb	.L83
	.p2align 4,,10
	.p2align 3
.L80:
	cmpq	%rsi, %r9
	jnb	.L95
	cmpq	%r9, %r14
	jbe	.L95
	leaq	(%rbx,%r8,4), %rdx
	movq	%r11, %rax
	movq	%rbp, %rcx
	vmovaps	%xmm1, %xmm0
	jmp	.L79
	.p2align 4,,10
	.p2align 3
.L121:
	addq	$32, %rax
	addq	$32, %rdx
	cmpq	%rcx, %r14
	jbe	.L77
.L79:
	vmovss	(%rax), %xmm2
	vmovss	4(%rax), %xmm3
	vfmadd231ss	(%rdx), %xmm2, %xmm0
	vmovss	8(%rax), %xmm4
	vmovss	12(%rax), %xmm5
	vmovss	16(%rax), %xmm6
	vmovss	20(%rax), %xmm7
	vfmadd231ss	4(%rdx), %xmm3, %xmm0
	vmovss	24(%rax), %xmm2
	vmovss	28(%rax), %xmm3
	addq	$8, %rcx
	vfmadd231ss	8(%rdx), %xmm4, %xmm0
	vfmadd231ss	12(%rdx), %xmm5, %xmm0
	vfmadd231ss	16(%rdx), %xmm6, %xmm0
	vfmadd231ss	20(%rdx), %xmm7, %xmm0
	vfmadd231ss	24(%rdx), %xmm2, %xmm0
	vfmadd231ss	28(%rdx), %xmm3, %xmm0
	cmpq	%rsi, %rcx
	jb	.L121
.L77:
	vaddss	(%rdi), %xmm0, %xmm0
	addq	$4, %rdi
	addq	%r14, %r8
	vmovss	%xmm0, -4(%rdi)
	cmpq	%rdi, %r10
	jne	.L80
.L83:
	incq	%r12
	addq	-88(%rsp), %r10
	addq	-96(%rsp), %r11
	cmpq	-112(%rsp), %r12
	jne	.L85
.L81:
	movq	32(%rsp), %rdi
	addq	256(%rsp), %r9
	addq	%rdi, 48(%rsp)
	addq	%rdi, %rbx
	cmpq	%rsi, 72(%rsp)
	ja	.L87
.L84:
	movq	8(%rsp), %rbx
	addq	%rbx, -120(%rsp)
	movq	32(%rsp), %rbx
	addq	%rbx, 40(%rsp)
	movq	64(%rsp), %rbx
	cmpq	%rbx, -40(%rsp)
	jb	.L89
.L86:
	movq	144(%rsp), %rbx
	addq	%rbx, 24(%rsp)
	movq	8(%rsp), %rbx
	addq	%rbx, 16(%rsp)
	movq	56(%rsp), %rbx
	cmpq	%rbx, -56(%rsp)
	jb	.L91
	movq	%r14, %r13
.L88:
	cmpq	-64(%rsp), %r13
	ja	.L93
.L90:
	movq	-24(%rsp), %rbx
	addq	%rbx, (%rsp)
	movq	-72(%rsp), %rbx
	cmpq	%rbx, -48(%rsp)
	ja	.L94
.L92:
	movq	112(%rsp), %rbx
	addq	%rbx, -8(%rsp)
	movq	-24(%rsp), %rbx
	addq	%rbx, -16(%rsp)
	movq	-80(%rsp), %rbx
	cmpq	%rbx, -32(%rsp)
	ja	.L76
.L119:
	addq	$192, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L95:
	.cfi_restore_state
	vmovaps	%xmm1, %xmm0
	jmp	.L77
	.cfi_endproc
.LFE5310:
	.size	l1_tiled_matmul, .-l1_tiled_matmul
	.p2align 4
	.globl	simd_kernel
	.type	simd_kernel, @function
simd_kernel:
.LFB5311:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	xorl	%r11d, %r11d
	xorl	%r10d, %r10d
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-32, %rsp
	subq	$416, %rsp
	movq	%fs:40, %rax
	movq	%rax, 408(%rsp)
	xorl	%eax, %eax
.L123:
	movl	%r10d, %eax
	addl	$32, %r10d
	movq	%r11, (%rsp,%rax)
	movq	%r11, 8(%rsp,%rax)
	movq	%r11, 16(%rsp,%rax)
	movq	%r11, 24(%rsp,%rax)
	cmpl	$384, %r10d
	jb	.L123
	testq	%r9, %r9
	je	.L130
	vxorps	%xmm13, %xmm13, %xmm13
	salq	$2, %rcx
	salq	$2, %r8
	vmovaps	%ymm13, %ymm12
	vmovaps	%ymm13, %ymm11
	vmovaps	%ymm13, %ymm10
	vmovaps	%ymm13, %ymm9
	vmovaps	%ymm13, %ymm8
	vmovaps	%ymm13, %ymm7
	vmovaps	%ymm13, %ymm6
	vmovaps	%ymm13, %ymm5
	vmovaps	%ymm13, %ymm4
	vmovaps	%ymm13, %ymm2
	vmovaps	%ymm13, %ymm3
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L126:
	vbroadcastss	(%rsi), %ymm14
	vmovups	(%rdi), %ymm1
	vmovups	32(%rdi), %ymm0
	vfmadd231ps	%ymm14, %ymm1, %ymm3
	vfmadd231ps	%ymm14, %ymm0, %ymm2
	vbroadcastss	4(%rsi), %ymm14
	incq	%rax
	vfmadd231ps	%ymm14, %ymm1, %ymm4
	vfmadd231ps	%ymm14, %ymm0, %ymm5
	addq	%rcx, %rdi
	vmovaps	%ymm4, 64(%rsp)
	vmovaps	%ymm5, 96(%rsp)
	vbroadcastss	8(%rsi), %ymm14
	vfmadd231ps	%ymm14, %ymm1, %ymm6
	vfmadd231ps	%ymm14, %ymm0, %ymm7
	vmovaps	%ymm6, 128(%rsp)
	vmovaps	%ymm7, 160(%rsp)
	vbroadcastss	12(%rsi), %ymm14
	vfmadd231ps	%ymm14, %ymm1, %ymm8
	vfmadd231ps	%ymm14, %ymm0, %ymm9
	vmovaps	%ymm8, 192(%rsp)
	vmovaps	%ymm9, 224(%rsp)
	vbroadcastss	16(%rsi), %ymm14
	vfmadd231ps	%ymm14, %ymm1, %ymm10
	vfmadd231ps	%ymm14, %ymm0, %ymm11
	vmovaps	%ymm10, 256(%rsp)
	vmovaps	%ymm11, 288(%rsp)
	vbroadcastss	20(%rsi), %ymm14
	addq	%r8, %rsi
	vfmadd231ps	%ymm14, %ymm1, %ymm12
	vfmadd231ps	%ymm14, %ymm0, %ymm13
	vmovaps	%ymm12, 320(%rsp)
	vmovaps	%ymm13, 352(%rsp)
	cmpq	%rax, %r9
	jne	.L126
.L125:
	movq	16(%rbp), %rax
	leaq	384(%rsp), %rsi
	leaq	0(,%rax,4), %rcx
	movq	%rsp, %rax
	jmp	.L128
	.p2align 4,,10
	.p2align 3
.L135:
	vmovaps	(%rax), %ymm3
	vmovaps	32(%rax), %ymm2
.L128:
	addq	$64, %rax
	vmovups	%ymm3, (%rdx)
	vmovups	%ymm2, 32(%rdx)
	addq	%rcx, %rdx
	cmpq	%rax, %rsi
	jne	.L135
	movq	408(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L136
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L130:
	.cfi_restore_state
	vxorps	%xmm2, %xmm2, %xmm2
	vmovaps	%ymm2, %ymm3
	jmp	.L125
.L136:
	vzeroupper
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE5311:
	.size	simd_kernel, .-simd_kernel
	.p2align 4
	.globl	simd_matmul
	.type	simd_matmul, @function
simd_matmul:
.LFB5312:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$56, %rsp
	.cfi_def_cfa_offset 112
	movq	%rsi, 24(%rsp)
	testq	%rcx, %rcx
	je	.L149
	movq	%r8, %rax
	salq	$6, %rax
	movq	%rax, 32(%rsp)
	movq	%rcx, %rax
	salq	$6, %rax
	movq	%rdx, 16(%rsp)
	movq	%rax, 40(%rsp)
	movq	$0, 8(%rsp)
	movq	%rdi, %rbx
	movq	%rcx, %rbp
	movq	%r8, %r15
	.p2align 4,,10
	.p2align 3
.L139:
	movq	24(%rsp), %r13
	movq	16(%rsp), %r14
	xorl	%r12d, %r12d
	testq	%r15, %r15
	je	.L142
	.p2align 4,,10
	.p2align 3
.L140:
	pushq	$6
	.cfi_def_cfa_offset 120
	movq	%r14, %rdx
	movq	%r13, %rsi
	pushq	$16
	.cfi_def_cfa_offset 128
	movq	%r15, %r8
	movq	%rbp, %rcx
	movq	%rbx, %rdi
	movq	%r9, 16(%rsp)
	addq	$6, %r12
	call	simd_kernel
	popq	%rax
	.cfi_def_cfa_offset 120
	addq	$24, %r13
	addq	$24, %r14
	popq	%rdx
	.cfi_def_cfa_offset 112
	cmpq	%r12, %r15
	movq	(%rsp), %r9
	ja	.L140
.L142:
	addq	$16, 8(%rsp)
	movq	32(%rsp), %rsi
	addq	40(%rsp), %rbx
	addq	%rsi, 16(%rsp)
	movq	8(%rsp), %rax
	cmpq	%rax, %rbp
	ja	.L139
.L149:
	addq	$56, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE5312:
	.size	simd_matmul, .-simd_matmul
	.ident	"GCC: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	 1f - 0f
	.long	 4f - 1f
	.long	 5
0:
	.string	 "GNU"
1:
	.align 8
	.long	 0xc0000002
	.long	 3f - 2f
2:
	.long	 0x3
3:
	.align 8
4:
