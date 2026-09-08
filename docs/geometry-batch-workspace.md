# Internal geometry-batch workspace

The algebraic full-curve FFI receives geometry, fixed LD and numerical controls;
it need not know event trajectories, parallax or Bayesian parameters. MOASARC
owns those transformations and chains the five-column geometry Jacobian.

The FFI retains one magnifier per execution worker, rebuilding it on LD changes.
Its existing lens-dependent invalidation remains authoritative. Mutable state
is thread-local; no shared model handle or assumed proposal sequence is needed.
The critical-event policy bypasses the historical caustic/router cache: retaining
the magnifier must not be described as critical-event warm-start acceleration.

A separate exact-bit-key cache retains at most 1,024 successful authoritative
value/Jacobian results per worker. At capacity it clears before insertion.
Keys contain x,y,rho,q,s,c,d,absolute/relative value tolerance, gradient tolerance
and refinement policy. Failure results are never retained. Forward and Jacobian
targets share the same result. Strict reporting is applied after lookup, so it
does not alter the numerical key. Finite uncertified results keep their status
and error estimates. Diagnostic work arrays describe the original solve, not
newly executed work on a cache hit.

Internal Python controls are `_lcbinint._jax_ir.geometry_cache_info()` and
`reset_geometry_cache(enabled=True)`. Disable for cold comparisons. Reset
increments a generation, invalidated lazily at each worker's next batch; an
in-flight batch retains its magnifier safely. Counter reset should follow a
synchronization when exact accounting is needed. Point rows are not memoized.

The cache is bounded and independent of model lifetimes. Input changes miss it;
it is not approximate continuation or a substitute for valid initial image
coverage. New engine workspace strategies can remain behind this FFI without
adding physical model arguments. Future varying FiniteSourceSettings fields
must extend magnifier invalidation and any numerical-key dependencies.

Connection regression tests live in the integration MOASARC checkout at
`tests/test_geometry_cache.py`; native full-curve contract tests remain in
`tests/jax_ir/test_algebraic_boundary_batch.py`.
