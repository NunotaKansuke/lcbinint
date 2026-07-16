from __future__ import annotations

import numpy as np
import pytest


class _GaussianPosterior:
    param_names = ["x"]
    optimizer_bounds = [(-8.0, 8.0)]
    _sample_transforms = ["identity"]

    @staticmethod
    def has_reparams():
        return False

    @staticmethod
    def has_py_extras():
        return True

    @staticmethod
    def validate():
        return None

    def _sampling_adapter(self):
        return self

    @staticmethod
    def n_params():
        return 1

    @staticmethod
    def log_prob(theta):
        return -0.5 * ((theta[0] - 1.25) / 0.7) ** 2

    @staticmethod
    def consume_current_aux(positions):
        nwalkers = len(positions)
        fluxes = np.zeros((nwalkers, 2), dtype=float)
        scales = np.zeros((nwalkers, 1), dtype=float)
        return fluxes, scales, np.zeros(1, dtype=float), ["dummy"]


class _ScalarLogProb:
    @staticmethod
    def log_prob(theta):
        return -0.5 * sum(value * value for value in theta)


class _BatchLogProb(_ScalarLogProb):
    def __init__(self):
        self.batch_sizes = []

    def log_prob_batch(self, rows):
        self.batch_sizes.append(len(rows))
        return [self.log_prob(theta) for theta in rows]


def test_batch_sampler_step_matches_scalar_rng_and_state():
    lcbinint = pytest.importorskip("lcbinint")

    nwalkers = 8
    ndim = 2
    seed = 19
    pos = np.linspace(-1.5, 1.5, nwalkers * ndim).reshape(nwalkers, ndim)
    log_prob = np.asarray([_ScalarLogProb.log_prob(row) for row in pos])
    scalar_state = lcbinint.sample._make_state(
        nwalkers, ndim, seed, pos.copy(), log_prob.copy()
    )
    batch_state = lcbinint.sample._make_state(
        nwalkers, ndim, seed, pos.copy(), log_prob.copy()
    )
    scalar_sampler = lcbinint.sample.EnsembleSampler(nwalkers, seed)
    batch_sampler = lcbinint.sample.EnsembleSampler(nwalkers, seed)
    batch_model = _BatchLogProb()

    for _ in range(5):
        scalar_sampler.step(_ScalarLogProb(), scalar_state)
        batch_sampler.step(batch_model, batch_state)

    np.testing.assert_array_equal(batch_state.pos, scalar_state.pos)
    np.testing.assert_array_equal(batch_state.log_prob, scalar_state.log_prob)
    np.testing.assert_array_equal(batch_state.get_chain, scalar_state.get_chain)
    np.testing.assert_array_equal(
        batch_state.get_log_prob, scalar_state.get_log_prob
    )
    assert batch_state.acceptance_fraction == scalar_state.acceptance_fraction
    assert batch_model.batch_sizes == [nwalkers // 2] * 10


def test_ensemble_sampler_recovers_a_known_gaussian_posterior():
    lcbinint = pytest.importorskip("lcbinint")

    chain = lcbinint.run_sampler(
        _GaussianPosterior(),
        burnin=500,
        nsteps=1000,
        start=[1.25],
        options=lcbinint.SamplerOptions(
            nwalkers=32,
            seed=47,
            log_path="",
            auto_stop=False,
        ),
    )
    samples = chain.get_samples()[:, 0]

    assert np.mean(samples) == pytest.approx(1.25, abs=0.06)
    assert np.std(samples, ddof=1) == pytest.approx(0.7, abs=0.06)
    assert 0.2 < chain.acceptance_fraction < 0.9
