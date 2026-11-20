# Source attribution and license

The captured `reproducer/base_kernel.py` and `reproducer/candidate_kernel.py` contain extracted `exp2` and `chunk_gla_fwd_kernel_o_tle` definitions from [FlagGems-vllm chunk_gla.py at14a3602284b86337ffcf9fa7d025f9ff89a411bb](https://github.com/flagos-ai/FlagGems-vllm/blob/14a3602284b86337ffcf9fa7d025f9ff89a411bb/src/flaggems_vllm/ops/FLA/chunk_gla.py). The original source carries this attribution:

```text
Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
For a list of all contributors, visit:
https://github.com/fla-org/flash-linear-attention/graphs/contributors
```

The upstream project is [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention). The complete MIT license text, with the same copyright holders, is preserved byte-for-byte in [LICENSE.MIT-FLA.txt](LICENSE.MIT-FLA.txt), verified against [official FLA v0.6.0 commit413ad2c655ece57b94bc9ddc51fdbd43cf09d984 LICENSE](https://github.com/fla-org/flash-linear-attention/blob/413ad2c655ece57b94bc9ddc51fdbd43cf09d984/LICENSE), SHA256 `41a83c8187efc1e3ccc21909e806a9e52338e69448554d9754706c3d1cd610e7`. This license source identifies the existing upstream MIT terms; the captured definitions were extracted from the FlagGems-vllm source above, not claimed to be a verbatim v0.6.0 implementation.

The candidate adds one constexpr and changes two static-range bounds, as shown by `reproducer/candidate.diff`. Extraction removes autotune/heuristic wrappers and enclosing module indentation, retains the JIT definition, and preserves kernel-body AST identity. This attribution is supplied at bundle level to preserve the captured modules' original evidence bytes: baseline module SHA256 `5a30db4a46ad5b9b40e7c5be89026c71752361f1275c821cbc04be7784be7697`, candidate module SHA256 `58c45ef148c253d0f0116fb790ed4dadbf2a47751b71a119a11815c842b114fc`. Those module files and the production diff have not been edited to add a header after capture.

Compiler-generated IR/PTX is included solely as diagnostic output of these attributed inputs and the pinned compiler. Other reproduction helpers and summaries describe the experiment; they do not replace or remove the upstream notice or license attached here.
