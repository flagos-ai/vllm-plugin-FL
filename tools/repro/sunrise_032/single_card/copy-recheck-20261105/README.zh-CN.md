# 2026-11-05：单卡验证完成后的0/2纯拷贝复查

27B TP1 eager/graph退出后，用[原执行脚本](../../current_environment/repro_tensor_roundtrip.py)对mask0、2各做一次全量拷贝检查，不运行模型、GEMM、转置、通信或graph。执行源SHA256均为`c9c96bf53575811c0bdadaf8dc64e9dfe1ed602c228bd558f43af23320738d0f`。

两卡各9次比较中，input5次均失败，weight4次精确；首坏runtime指针地址仍为`0xfff3bd3c8`。CPU源未变，worker正常exit0，无timeout，CLI因数值错误返回1。地址不是物理DRAM定位或已证实的DMA根因。[卡0原始记录](card0/roundtrip.jsonl)、[卡2原始记录](card2/roundtrip.jsonl)与summary保持原字节；来源见[provenance](provenance.json)。

[最终状态](final-state.json)仍见0/2/4为0B/0%，1/3/5/6原有1.535GiB/8%，7原有12KiB/8%，无列出GPU进程，表中未见新增残留。空闲表不等于数值健康。未reset、重启或卸载驱动；GPU验证已结束。

剩余TP2需要同一台机器上两张基础数值及通信均可靠、直连且可独占的卡，不要求第二台机器。应由资源管理员和曦望侧确认故障上下文恢复方式及peer影响范围；现在没有证明整机重启必要或足以修复的证据。
