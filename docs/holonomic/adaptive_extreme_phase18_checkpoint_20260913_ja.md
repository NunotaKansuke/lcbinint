# Phase18: fixed-degree D14 convolution

Phase17 a199c58を基準に、係数展開のzero workを調査。
HOLO_D14_DIRECT_CONVOLUTION研究flagでは、出力係数ごとの畳み込みにし、
最初の積を直接accumulatorへ代入する。以後は従来のascending-i加算順。
係数精度・求根・certificate・production defaultは変更しない。

7216入力、各21回の中央値でmicrobench:
旧 p50/p90/p99 = 5.652/5.684/6.101 us、
候補 = 5.031/5.076/5.192 us。
全係数をqf hexadecimalでexportし、全行一致。
約0.6usの節約なのでwhole勝利とはしない。

再現: bench_d14_expansion.cppを-O3 -DNDEBUG -std=gnu++17
-fext-numeric-literals -Isrc -march=nativeでcompileし、flag有無で比較。
input_snapshot.tsvを渡しCPU0で逐次実行。
初回compileはre_detail namespace修飾漏れで失敗し修正済み。

次にPhase17 screenと併用したwhole runnerを同じflags/inputs/repeat3で測る。
Phase17 whole_screen.tsvをbaselineとし、全費用込みで判定する。
現段階では研究flag、whole未確認。

既存test_d14_root_schedulingを候補flag付きでcompile/runしpass。
全epoch計測のCPU0を避け、この短いcorrectness確認のみCPU1で実行。

## Whole完了・判断

全14,432行、RelTol各7216行で全converged。mu/status/node差0。
1e-3 p50 cold .414842→.414029 ms、warm .341652→.340798 ms。
1e-4 p50 cold .458476→.458364 ms、warm .3901425→.3896575 ms。
1e-3 p99 cold 3.192342→3.210154 ms、warm 1.138409→1.146416 ms。
radial-only p50 .121088→.121538 ms（1e-3）。

microの約0.6us短縮と整合する小さいp50差だが、wholeの測定変動から
十分分離できていない。production defaultは変更せず研究flagを保持。
VBMに対するwhole勝利とは主張しない。追加の繰り返しでこの小差だけを
追うより、次はendpoint/geometryのより大きい仕事を減らす。
VBM reference超過数は1e-3=0、1e-4=既存3で不変。

whole再現: Phase17のreproduce.sh中whole compileに
-DHOLO_D14_DIRECT_CONVOLUTIONを追加。その他flags/input/repeat3/CPU0同一。
summaryはsummarize_adaptive_reciprocal.pyへ
--baseline ../adaptive_extreme_phase17/whole_screen.tsv
--candidate ../adaptive_extreme_phase18/whole_candidate.tsv
--output ../adaptive_extreme_phase18/whole_summary.json。
