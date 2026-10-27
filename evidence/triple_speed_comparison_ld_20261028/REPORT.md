# Triple direct integration timing

CPU0 thread_time_ns; 4 warmups and 16 rotating/reversed timed calls. Persistent separate state per condition and engine; geometry setup excluded; lcbinint includes support preparation and integral; VBM direct MultiMag/MultiMagDark Tol=0 a2=0. Fresh normal-router screen before timing. No light-curve, hex or source-disc timing. Tight references are saved prior independent VBM runs, diagnostic only.

One lens geometry, 12 source disks, 48 conditions. Linear LD coefficient u=0.5. Each cell below is the median across disks of each disk’s 16-call median; milliseconds. Ratios are computed per disk before aggregation. CPU frequency and unrelated system load are uncontrolled. VBM columns are value only.

| u | rtol | SinglePoly | MultiPoly | NoPoly | lcbinint value | lcbinint value+8 derivatives | median derivative overhead |
|---|---|---|---|---|---|---|---|
|0.0|0.001|0.673|0.864|2.052|0.785|1.104|52.3%|
|0.0|0.0001|1.226|1.572|3.703|0.926|1.253|38.0%|
|0.5|0.001|12.626|16.608|38.908|1.066|1.894|86.5%|
|0.5|0.0001|57.036|75.192|177.744|1.248|2.121|62.8%|

## Checks

{
  "conditions": 48,
  "excluded": 0,
  "methods": [
    2
  ],
  "native_nonconverged": [],
  "gradient_invalid": [],
  "above_requested": [],
  "max_relative_difference": 0.0003163367995992844
}

Every condition was freshly screened through the current normal finite-source router (method 2 = image-plane Cartesian integration). No point/hex cases were admitted. Direct integration, not light-curve API, is timed. Repeated-point warm performance includes retained support caches; first calls are retained separately in results.json.

Accuracy comparison uses the saved independent SinglePoly/MultiPoly/NoPoly tight runs at rtol=1e-7 for the identical inputs. These references are diagnostics, not certified truth; the previous reference audit found inter-engine spread up to 9.12e-6. No new derivative finite-difference certification was performed for these 12 disks; gradient_valid indicates usable output, not certified precision. The existing separate gradient regression suite remains the accuracy evidence.

## Per-condition timing

| x | y | rho | u | rtol | SinglePoly | MultiPoly | NoPoly | value | value+derivatives |
|---|---|---|---|---|---|---|---|---|---|
|0.07|0.15|0.075|0.0|0.001|0.721|0.907|2.181|1.499|1.694|
|0.07|0.15|0.075|0.0|0.0001|1.326|1.723|4.157|3.142|1.757|
|0.07|0.15|0.075|0.5|0.001|9.411|12.415|29.497|1.634|2.125|
|0.07|0.15|0.075|0.5|0.0001|41.195|53.878|129.620|4.773|2.356|
|0.1|0.15|0.075|0.0|0.001|0.810|1.057|2.512|1.072|1.396|
|0.1|0.15|0.075|0.0|0.0001|1.506|1.985|4.764|1.101|1.470|
|0.1|0.15|0.075|0.5|0.001|11.369|14.900|35.448|1.100|1.794|
|0.1|0.15|0.075|0.5|0.0001|44.660|59.406|140.831|1.207|1.958|
|0.13|0.15|0.075|0.0|0.001|0.817|1.045|2.517|1.092|1.337|
|0.13|0.15|0.075|0.0|0.0001|1.462|1.925|4.503|1.165|1.408|
|0.13|0.15|0.075|0.5|0.001|9.235|12.220|29.124|1.117|1.641|
|0.13|0.15|0.075|0.5|0.0001|37.482|49.677|118.617|1.289|1.828|
|0.1|0.18|0.075|0.0|0.001|0.651|0.850|2.007|0.883|1.052|
|0.1|0.18|0.075|0.0|0.0001|1.189|1.568|3.713|0.955|1.118|
|0.1|0.18|0.075|0.5|0.001|8.642|11.845|27.181|0.960|2.071|
|0.1|0.18|0.075|0.5|0.0001|33.538|45.066|102.240|1.121|2.204|
|0.07|0.15|0.15|0.0|0.001|0.703|0.915|2.115|0.593|0.961|
|0.07|0.15|0.15|0.0|0.0001|1.301|1.700|3.936|0.698|1.022|
|0.07|0.15|0.15|0.5|0.001|11.093|14.566|34.395|0.761|1.597|
|0.07|0.15|0.15|0.5|0.0001|60.677|79.929|189.072|1.203|1.966|
|0.1|0.15|0.15|0.0|0.001|0.644|0.840|1.946|0.612|0.956|
|0.1|0.15|0.15|0.0|0.0001|1.158|1.537|3.532|0.755|1.008|
|0.1|0.15|0.15|0.5|0.001|14.308|18.744|43.991|0.788|1.494|
|0.1|0.15|0.15|0.5|0.0001|58.189|76.321|180.421|0.936|1.675|
|0.13|0.15|0.15|0.0|0.001|0.600|0.791|1.772|0.558|0.827|
|0.13|0.15|0.15|0.0|0.0001|1.154|1.509|3.429|0.612|0.899|
|0.13|0.15|0.15|0.5|0.001|12.009|15.819|36.947|0.645|1.183|
|0.13|0.15|0.15|0.5|0.0001|55.883|74.165|175.136|0.786|1.358|
|0.1|0.18|0.15|0.0|0.001|0.679|0.877|2.094|0.732|1.168|
|0.1|0.18|0.15|0.0|0.0001|1.256|1.628|3.864|0.780|1.246|
|0.1|0.18|0.15|0.5|0.001|13.244|17.397|40.869|0.871|2.076|
|0.1|0.18|0.15|0.5|0.0001|52.328|69.539|160.586|0.953|2.297|
|0.07|0.15|0.3|0.0|0.001|0.601|0.771|1.748|0.837|1.528|
|0.07|0.15|0.3|0.0|0.0001|1.187|1.517|3.499|0.974|1.985|
|0.07|0.15|0.3|0.5|0.001|14.699|19.422|45.860|1.152|3.074|
|0.07|0.15|0.3|0.5|0.0001|68.895|90.001|214.916|2.145|4.154|
|0.1|0.15|0.3|0.0|0.001|0.648|0.825|1.901|0.683|1.075|
|0.1|0.15|0.3|0.0|0.0001|1.227|1.574|3.686|0.897|1.394|
|0.1|0.15|0.3|0.5|0.001|16.976|22.196|52.720|1.136|2.308|
|0.1|0.15|0.3|0.5|0.0001|72.333|95.886|220.837|3.818|3.046|
|0.13|0.15|0.3|0.0|0.001|0.667|0.851|2.009|1.550|1.077|
|0.13|0.15|0.3|0.0|0.0001|1.226|1.570|3.694|2.544|1.261|
|0.13|0.15|0.3|0.5|0.001|16.430|21.583|50.133|3.907|1.995|
|0.13|0.15|0.3|0.5|0.0001|67.181|91.978|215.201|10.293|2.537|
|0.1|0.18|0.3|0.0|0.001|0.690|0.881|2.100|0.679|1.131|
|0.1|0.18|0.3|0.0|0.0001|1.106|1.328|3.274|0.737|1.051|
|0.1|0.18|0.3|0.5|0.001|14.214|18.785|43.484|1.031|1.764|
|0.1|0.18|0.3|0.5|0.0001|58.674|76.219|180.352|1.516|2.038|

## Matched optimization improvement

Two separate runs of four warmups and sixteen rotated/reversed samples compare the final guarded binary against the immediately preceding validated working-tree binary. These are pointwise relative changes, not differences between historical timing tables.

| u | rtol | value time reduction | value+derivatives time reduction | before added cost | after added cost |
|---|---|---|---|---|---|
|0.0|0.001|-0.58–-0.56%|0.50–0.59%|54.19–55.45%|51.57–51.73%|
|0.0|0.0001|-0.82–-0.34%|0.94–1.26%|42.16–43.07%|39.94–40.75%|
|0.5|0.001|-0.71–0.12%|4.26–4.46%|100.98–101.49%|89.31–89.57%|
|0.5|0.0001|1.37–1.48%|4.89–5.30%|70.07–70.33%|63.23–64.20%|

The scalar LD rtol1e-3 result is effectively unchanged. Scalar LD rtol1e-4 improves about1.4%; gradient LD improves about4–5%. Uniform timings are approximately unchanged. The 30% gradient-overhead goal is not reached.

Final checks: CTest30/30; scalar168/168; independent scalar FD160 canonical components plus80 pure-LD identity components pass the unchanged derivative gate, including reference uncertainty; physical gradients88/88 through both C ABI and rebuilt public Python API; broad comparator120/120, saved86/86 and new holdout24/24. A newly found strict small-source value regression was repaired and fixed in test_regular.cpp; three previously converged strict requests regain baseline quad counts. The existing tighter rho=.001,rtol1e-9 baseline failure is preserved as an out-of-scope diagnostic, not claimed solved.

Accepted changes: retain all endpoint/map uncertainty; use a weighted absolute roundoff budget for loose scalar samples, with incumbent early local promotion for strict requests; share mirrored angular sine/cosine and scalar contractions across eight derivatives; reject absent coordinate-hint maxima before a cubic solve. Implementation and rejected alternatives are recorded in evidence/triple_ld_final_cost/NOTES.md. No numerical tolerance was relaxed. No source-disc, point/hex, or light-curve timing is included.
