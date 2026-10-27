# Triple direct integration timing

CPU0 thread_time_ns; 4 warmups and 16 rotating/reversed timed calls. Persistent separate state per condition and engine; geometry setup excluded; lcbinint includes support preparation and integral; VBM direct MultiMag/MultiMagDark Tol=0 a2=0. Fresh normal-router screen before timing. No light-curve, hex or source-disc timing. Tight references are saved prior independent VBM runs, diagnostic only.

One lens geometry, 12 source disks, 48 conditions. Linear LD coefficient u=0.5. Each cell below is the median across disks of each disk’s 16-call median; milliseconds. Ratios are computed per disk before aggregation. CPU frequency and unrelated system load are uncontrolled. VBM columns are value only.

| u | rtol | SinglePoly | MultiPoly | NoPoly | lcbinint value | lcbinint value+8 derivatives | median derivative overhead |
|---|---|---|---|---|---|---|---|
|0.0|0.001|0.669|0.865|2.063|0.727|1.186|51.0%|
|0.0|0.0001|1.229|1.596|3.688|0.872|1.299|41.4%|
|0.5|0.001|12.809|16.935|39.457|1.095|2.063|70.4%|
|0.5|0.0001|58.172|76.910|180.554|1.304|2.270|58.1%|

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
|0.07|0.15|0.075|0.0|0.001|0.707|0.909|2.182|1.505|1.695|
|0.07|0.15|0.075|0.0|0.0001|1.332|1.733|4.203|3.183|1.768|
|0.07|0.15|0.075|0.5|0.001|9.242|12.158|28.932|1.620|2.125|
|0.07|0.15|0.075|0.5|0.0001|40.349|51.337|122.181|4.785|2.348|
|0.1|0.15|0.075|0.0|0.001|0.810|1.050|2.513|1.073|1.411|
|0.1|0.15|0.075|0.0|0.0001|1.478|1.923|4.665|1.080|1.445|
|0.1|0.15|0.075|0.5|0.001|10.929|14.304|34.061|1.050|1.708|
|0.1|0.15|0.075|0.5|0.0001|42.528|56.970|136.040|1.177|1.907|
|0.13|0.15|0.075|0.0|0.001|0.654|0.841|2.030|0.888|1.091|
|0.13|0.15|0.075|0.0|0.0001|1.147|1.491|3.544|0.920|1.126|
|0.13|0.15|0.075|0.5|0.001|8.882|11.670|27.845|1.141|1.607|
|0.13|0.15|0.075|0.5|0.0001|35.734|47.038|111.472|1.286|1.808|
|0.1|0.18|0.075|0.0|0.001|0.497|0.650|1.530|0.683|0.815|
|0.1|0.18|0.075|0.0|0.0001|0.944|1.220|2.892|0.767|0.894|
|0.1|0.18|0.075|0.5|0.001|8.458|11.391|26.051|0.907|2.001|
|0.1|0.18|0.075|0.5|0.0001|33.419|44.544|100.825|1.123|2.193|
|0.07|0.15|0.15|0.0|0.001|0.710|0.920|2.128|0.596|0.973|
|0.07|0.15|0.15|0.0|0.0001|1.253|1.635|3.801|0.678|0.999|
|0.07|0.15|0.15|0.5|0.001|11.276|14.901|34.670|0.888|1.780|
|0.07|0.15|0.15|0.5|0.0001|61.628|79.481|189.476|1.322|2.067|
|0.1|0.15|0.15|0.0|0.001|0.648|0.849|1.945|0.630|0.959|
|0.1|0.15|0.15|0.0|0.0001|1.154|1.523|3.464|0.737|0.999|
|0.1|0.15|0.15|0.5|0.001|14.438|19.040|44.786|0.924|1.594|
|0.1|0.15|0.15|0.5|0.0001|59.140|78.198|182.413|1.087|1.738|
|0.13|0.15|0.15|0.0|0.001|0.597|0.777|1.759|0.558|0.835|
|0.13|0.15|0.15|0.0|0.0001|1.114|1.470|3.291|0.587|0.879|
|0.13|0.15|0.15|0.5|0.001|12.057|15.943|37.242|0.743|1.249|
|0.13|0.15|0.15|0.5|0.0001|57.204|75.622|178.695|0.882|1.465|
|0.1|0.18|0.15|0.0|0.001|0.684|0.881|2.096|0.737|1.188|
|0.1|0.18|0.15|0.0|0.0001|1.266|1.639|3.887|0.789|1.262|
|0.1|0.18|0.15|0.5|0.001|13.561|17.927|41.671|0.992|2.159|
|0.1|0.18|0.15|0.5|0.0001|53.157|70.828|163.140|1.093|2.450|
|0.07|0.15|0.3|0.0|0.001|0.610|0.778|1.776|0.855|1.940|
|0.07|0.15|0.3|0.0|0.0001|1.205|1.557|3.575|0.982|2.009|
|0.07|0.15|0.3|0.5|0.001|15.123|19.802|46.932|1.310|3.629|
|0.07|0.15|0.3|0.5|0.0001|71.262|93.599|220.971|2.281|4.328|
|0.1|0.15|0.3|0.0|0.001|0.631|0.795|1.825|0.668|1.183|
|0.1|0.15|0.3|0.0|0.0001|1.188|1.521|3.547|0.860|1.348|
|0.1|0.15|0.3|0.5|0.001|16.695|22.008|52.278|1.171|2.495|
|0.1|0.15|0.3|0.5|0.0001|75.242|99.292|231.797|4.142|3.338|
|0.13|0.15|0.3|0.0|0.001|0.703|0.900|2.129|1.659|1.280|
|0.13|0.15|0.3|0.0|0.0001|1.314|1.690|3.956|2.755|1.364|
|0.13|0.15|0.3|0.5|0.001|17.586|23.309|53.596|4.266|2.392|
|0.13|0.15|0.3|0.5|0.0001|75.455|99.283|228.988|11.029|2.936|
|0.1|0.18|0.3|0.0|0.001|0.725|0.919|2.185|0.716|1.212|
|0.1|0.18|0.3|0.0|0.0001|1.300|1.671|3.956|0.884|1.336|
|0.1|0.18|0.3|0.5|0.001|17.639|23.207|54.106|1.421|2.296|
|0.1|0.18|0.3|0.5|0.0001|74.520|98.032|225.741|2.050|2.706|

## Code provenance

Integrator source commit: `5ac69214b2128338866d389b80afd90aa95268b3`. Current post-removal C ABI kernel and a newly linked router were used. VBM 5.5 library hash matches the previous comparison; all three methods are explicit and value-only. Compiler jobs and validation runs completed before this timing run. The 30 native tests and separate 473-component offline derivative checks passed before timing. Source changes and unrelated JAX work were unchanged during measurement.

The derivative-overhead column is the median of per-disk ratios, not the ratio of the two aggregate timing medians. The 30% overhead target is not met. No speedup over the preceding build is inferred from separate-session absolute times.
