import hashlib
import os
import json
import random
import socket
import threading
import time
import traceback
from contextlib import contextmanager

import numpy as np
import inspect

from jmetal.problem import Sphere
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.reinitialization.reinitialized_pso import FRAPSO, PartialResetPSO, CollectiveResetPSO
from algorithm.reinitialization.boundary_reinitialized_pso import BoundaryReinitializedPSO

from algorithm.role_based.role_hybrids import (
    HybridPartialDisjointRestarterPSO,
    HybridFullDisjointRestarterPSO,
    HybridAdditiveRestarterPSO,
    HybridFullDisjointPSO_WithRandom,
    HybridPartialDisjointPSO_WithRandom,
    HybridAdditivePSO_WithRandom,
    HybridFullDisjointPSO,
    HybridPartialDisjointPSO,
    HybridAdditivePSO,
    HybridDisjointPSO_WithWanderer,
    HybridAdditivePSO_WithWanderer,
)
from algorithm.role_based.worst_aware_pso import (
    ReverseLearningPSO,
    ReverseLearningGlobalAttractorPSO,
    ReverseLearningPersonalAttractorPSO,
    CombinedLearningPSO,
)
from algorithm.role_based.adaptive_pso import CoAdaptativePSO as CAPSO, IndividualAdaptivePSO as IAPSO
from algorithm.pso_ga_hybrids.pgshea import PGSHEA
from algorithm.pso_ga_hybrids.pgphea import PGPHEA
from algorithm.basic.single_objective_pso import SingleObjectivePSO as PSO, PerturbationPSO
from algorithm.basic.differential_evolution import DifferentialEvolution
from algorithm.basic.custom_ga import GeneticAlgorithm
from algorithm.role_based.roles import (
    RebelPSO,
    RejectorPSO,
    ContrarianPSO,
    DefeatistPSO,
    EschewerPSO,
    EscapistPSO,
    AnarchicPSO,
    AmnesiacPSO,
    ErraticPSO,
    WandererPSO,
    RebelRejectorPSO,
    ContrarianDefeatistPSO,
    EschewerEscapistPSO,
    RRAPSO,
    CDAPSO,
    EEAPSO,
    AnarchicAmnesiacPSO,
    AAAPSO,
    NAPSO,
    CLAPSO,
    DrifterPSO,
    DAPSO,
)
from algorithm.sota.cma_es import CMAES
from algorithm.sota.lshade import LSHADE
from algorithm.pso_ga_hybrids.pgchea import PGCHEA
from jmetal.operator.crossover import SBXCrossover, DifferentialEvolutionCrossover
from jmetal.operator.mutation import PolynomialMutation

from algorithm.sparse_roles.sparse_hybrid import (
    SparseHybridAdditivePSO,
    SparseHybridFullDisjointPSO,
    SparseHybridPartialDisjointPSO,
)
from algorithm.sparse_roles.sparse_role_based import (
    SparseAmnesiacPSO,
    SparseAnarchicAmnesiacPSO,
    SparseAnarchicPSO,
    SparseContrarianDefeatistPSO,
    SparseContrarianPSO,
    SparseDefeatistPSO,
    SparseDrifterPSO,
    SparseErraticPSO,
    SparseEscapistPSO,
    SparseEschewerEscapistPSO,
    SparseEschewerPSO,
    SparseRebelPSO,
    SparseRebelRejectorPSO,
    SparseRejectorPSO,
    SparseWandererPSO,
)
from irace import irace, ParameterSpace, Scenario, Experiment, Real, Integer, Bool, Categorical
import rpy2.robjects as robjects

from problem.n_variables.ackley import Ackley
from problem.n_variables.CEC import ShiftedRotatedRastrigin
from experiment.problem_identity import create_seeded_problem

os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["R_DEFAULT_ENCODING"] = "UTF-8"
robjects.r('Sys.setlocale("LC_ALL", "en_US.UTF-8")')
# robjects.r('library(iraceplot)')


# number_of_variables = 10
# solutions_size = 10
# max_evaluations = 1000
# num_runs = 2
# budget = 60


from experiment.globals import G_SOLUTIONS_SIZE

number_of_variables = 100
solutions_size = G_SOLUTIONS_SIZE
# Truncated tuning horizon: 250k evals = 2,500 generations per run,
# deliberately below the 10^4 * D deployment budget (see docs/DISCLOSURES.md).
max_evaluations = 250_000
num_runs = 3  # Number of independent runs per problem
budget = 1000  # irace experiments (config-instance evaluations) per parameter

# Tuning instances. The first three are unrotated/(near-)separable classics;
# the seeded rotated Rastrigin adds a fully coupled non-separable landscape so
# racing also judges configurations on variable interactions. NONE of these may
# appear in the evaluation suite (no tuning/test leakage) - in particular,
# ShiftedRotatedRastrigin must stay out of the final suite roster.
TUNING_INSTANCE_SEED = 1042  # deliberately distinct from BENCHMARK_BASE_SEED
problems = [
    Sphere(number_of_variables),
    Rastrigin(number_of_variables),
    Ackley(number_of_variables),
    create_seeded_problem(ShiftedRotatedRastrigin, number_of_variables, TUNING_INSTANCE_SEED),
]


def base_pso_params():
    return [
        Real("w", 0.01, 1.0),
        Real("c1", 0.01, 6.0),
        Real("c2", 0.01, 6.0),
    ]


def single_sparse_mask_params():
    # coordinate_mode is fixed to "fraction" (dimension-invariant transfer);
    # injected by repair_and_normalize_config, not tuned.
    return [
        Real("coordinate_fraction", 0.0, 1.0),
    ]


def component_sparse_mask_params():
    return [
        Real("social_coordinate_fraction", 0.0, 1.0),
        Real("cognitive_coordinate_fraction", 0.0, 1.0),
    ]


def sparse_single_role_params(coefficient_name: str, fraction_name: str):
    return [
        *base_pso_params(),
        Real(coefficient_name, 0.01, 6.0),
        Real(fraction_name, 0.01, 0.99),
        *single_sparse_mask_params(),
    ]


parameter_spaces = {

        'BoundaryReinitializedPSO': {
            'params': [
                *base_pso_params(),
                Real("pbest_gbest_epsilon", 1e-4, 0.5, log=True),
                Categorical("distance_metric", ["normalized_rms", "normalized_linf", "fraction_close"]),
                Categorical("boundary_strategy", ["random_face", "near_boundary", "mixed_boundary"]),
                Real("boundary_margin", 0.01, 0.3),
                Categorical("velocity_reset_strategy", ["zero", "random", "away_from_gbest"]),
                Real("velocity_scale", 0.01, 0.5),
                Bool("reset_personal_best_on_reinit"),
            ],
        },

        'HybridPartialDisjointRestarterPSO': {
            'params': [
                *base_pso_params(),
                Real("rejector_c", 0.01, 6.0),
                Real("defeatist_c", 0.01, 6.0),
                Real("escapist_c", 0.01, 6.0),
                Real("amnesiac_c", 0.01, 6.0),
                Real("rebel_c", 0.01, 6.0),
                Real("contrarian_c", 0.01, 6.0),
                Real("eschewer_c", 0.01, 6.0),
                Real("anarchic_c", 0.01, 6.0),
                Real("rejector_fraction", 0.01, 0.99),
                Real("defeatist_fraction", 0.01, 0.99),
                Real("escapist_fraction", 0.01, 0.99),
                Real("amnesiac_fraction", 0.01, 0.99),
                Real("rebel_fraction", 0.01, 0.99),
                Real("contrarian_fraction", 0.01, 0.99),
                Real("eschewer_fraction", 0.01, 0.99),
                Real("anarchic_fraction", 0.01, 0.99),
                Bool("assign_roles_every_iteration"),
                Real("restarter_fraction", 0.01, 0.99),
                Real("convergence_threshold", 1e-3, 0.5, log=True),
            ],
        },

        'HybridFullDisjointRestarterPSO': {
            'params': [
                *base_pso_params(),
                Real("rejector_c", 0.01, 6.0),
                Real("defeatist_c", 0.01, 6.0),
                Real("escapist_c", 0.01, 6.0),
                Real("amnesiac_c", 0.01, 6.0),
                Real("rebel_c", 0.01, 6.0),
                Real("contrarian_c", 0.01, 6.0),
                Real("eschewer_c", 0.01, 6.0),
                Real("anarchic_c", 0.01, 6.0),
                Real("rejector_fraction", 0.01, 0.99),
                Real("defeatist_fraction", 0.01, 0.99),
                Real("escapist_fraction", 0.01, 0.99),
                Real("amnesiac_fraction", 0.01, 0.99),
                Real("rebel_fraction", 0.01, 0.99),
                Real("contrarian_fraction", 0.01, 0.99),
                Real("eschewer_fraction", 0.01, 0.99),
                Real("anarchic_fraction", 0.01, 0.99),
                Bool("assign_roles_every_iteration"),
                Real("restarter_fraction", 0.01, 0.99),
                Real("convergence_threshold", 1e-3, 0.5, log=True),
            ],
        },

        'HybridAdditiveRestarterPSO': {
            'params': [
                *base_pso_params(),
                Real("rejector_c", 0.01, 6.0),
                Real("defeatist_c", 0.01, 6.0),
                Real("escapist_c", 0.01, 6.0),
                Real("amnesiac_c", 0.01, 6.0),
                Real("rebel_c", 0.01, 6.0),
                Real("contrarian_c", 0.01, 6.0),
                Real("eschewer_c", 0.01, 6.0),
                Real("anarchic_c", 0.01, 6.0),
                Real("std_cognitive_prob", 0.01, 0.99),
                Real("rejector_prob", 0.01, 0.99),
                Real("defeatist_prob", 0.01, 0.99),
                Real("escapist_prob", 0.01, 0.99),
                Real("amnesiac_prob", 0.01, 0.99),
                Real("std_social_prob", 0.01, 0.99),
                Real("rebel_prob", 0.01, 0.99),
                Real("contrarian_prob", 0.01, 0.99),
                Real("eschewer_prob", 0.01, 0.99),
                Real("anarchic_prob", 0.01, 0.99),
                Bool("assign_flags_every_iteration"),
                Real("restarter_fraction", 0.01, 0.99),
                Real("convergence_threshold", 1e-3, 0.5, log=True),
            ],
        },

        'PGCHEA': {
            'params': [
                *base_pso_params(),
                Categorical("starting_algorithm", ["PSO", "GA"]),
                Bool("inherit_best"),
                Real("crossover_probability", 0.6, 1.0),
                Real("sbx_distribution_index", 2.0, 30.0, log=True),
                Real("mutation_distribution_index", 5.0, 100.0, log=True),
            ],
        },

        'FRAPSO': {
            'params': [
                *base_pso_params(),
                Integer("fractal_depth", 1, 6),
                Real("convergence_threshold", 1e-3, 0.5, log=True),
            ],
        },

        'PartialResetPSO': {
            'params': [
                *base_pso_params(),
                Real("convergence_threshold", 1e-3, 0.5, log=True),
                Real("restarter_fraction", 0.01, 0.99),
            ],
        },

        'CollectiveResetPSO': {
            'params': [
                *base_pso_params(),
                Real("convergence_threshold", 1e-3, 0.5, log=True),
            ],
        },

        'SparseWandererPSO': {
            'params': [
                *base_pso_params(),
                Real("noise_strength", 0.01, 3.0),
                Real("wanderer_fraction", 0.01, 0.99),
                *single_sparse_mask_params(),
            ],
        },

        'SparseDefeatistPSO': {
            'params': sparse_single_role_params("defeatist_c", "defeatist_fraction"),
        },

        'SparseRebelPSO': {
            'params': sparse_single_role_params("rebel_c", "rebel_fraction"),
        },

        'SparseRejectorPSO': {
            'params': sparse_single_role_params("rejector_c", "rejector_fraction"),
        },

        'SparseContrarianPSO': {
            'params': sparse_single_role_params("contrarian_c", "contrarian_fraction"),
        },

        'SparseEschewerPSO': {
            'params': sparse_single_role_params("eschewer_c", "eschewer_fraction"),
        },

        'SparseEscapistPSO': {
            'params': sparse_single_role_params("escapist_c", "escapist_fraction"),
        },

        'SparseAnarchicPSO': {
            'params': [
                *base_pso_params(),
                Real("random_strength", 0.01, 6.0),
                Real("anarchic_fraction", 0.01, 0.99),
                *single_sparse_mask_params(),
            ],
        },

        'SparseAmnesiacPSO': {
            'params': [
                *base_pso_params(),
                Real("random_strength", 0.01, 6.0),
                Real("amnesiac_fraction", 0.01, 0.99),
                *single_sparse_mask_params(),
            ],
        },

        'SparseErraticPSO': {
            'params': [
                *base_pso_params(),
                Real("random_strength", 0.01, 6.0),
                Real("erratic_fraction", 0.01, 0.99),
                *single_sparse_mask_params(),
            ],
        },

        'SparseDrifterPSO': {
            'params': [
                *base_pso_params(),
                Real("drifter_fraction", 0.01, 0.99),
                Real("perturbation_scale", 0.0001, 0.1, log=True),
                Categorical("perturbation_method", ["gaussian", "cauchy"]),
                *single_sparse_mask_params(),
            ],
        },

        'SparseContrarianDefeatistPSO': {
            'params': [
                *base_pso_params(),
                Real("defeatist_c", 0.01, 6.0),
                Real("contrarian_c", 0.01, 6.0),
                Real("contrarian_fraction", 0.01, 0.99),
                Real("defeatist_fraction", 0.01, 0.99),
                *component_sparse_mask_params(),
            ],
        },

        'SparseRebelRejectorPSO': {
            'params': [
                *base_pso_params(),
                Real("rejector_c", 0.01, 6.0),
                Real("rebel_c", 0.01, 6.0),
                Real("rebel_fraction", 0.01, 0.99),
                Real("rejector_fraction", 0.01, 0.99),
                *component_sparse_mask_params(),
            ],
        },

        'SparseEschewerEscapistPSO': {
            'params': [
                *base_pso_params(),
                Real("escapist_c", 0.01, 6.0),
                Real("eschewer_c", 0.01, 6.0),
                Real("eschewer_fraction", 0.01, 0.99),
                Real("escapist_fraction", 0.01, 0.99),
                *component_sparse_mask_params(),
            ],
        },

        'SparseAnarchicAmnesiacPSO': {
            'params': [
                *base_pso_params(),
                Real("random_strength_social", 0.01, 6.0),
                Real("random_strength_cognitive", 0.01, 6.0),
                Real("anarchic_fraction", 0.01, 0.99),
                Real("amnesiac_fraction", 0.01, 0.99),
                *component_sparse_mask_params(),
            ],
        },

        'SparseHybridPartialDisjointPSO': {
            'params': [
                *base_pso_params(),
                Real("rejector_c", 0.01, 6.0),
                Real("defeatist_c", 0.01, 6.0),
                Real("escapist_c", 0.01, 6.0),
                Real("rebel_c", 0.01, 6.0),
                Real("contrarian_c", 0.01, 6.0),
                Real("eschewer_c", 0.01, 6.0),
                Real("rejector_fraction", 0.01, 0.99),
                Real("defeatist_fraction", 0.01, 0.99),
                Real("escapist_fraction", 0.01, 0.99),
                Real("rebel_fraction", 0.01, 0.99),
                Real("contrarian_fraction", 0.01, 0.99),
                Real("eschewer_fraction", 0.01, 0.99),
                Bool("assign_roles_every_iteration"),
                *component_sparse_mask_params(),
            ],
        },

        'SparseHybridFullDisjointPSO': {
            'params': [
                *base_pso_params(),
                Real("rejector_c", 0.01, 6.0),
                Real("defeatist_c", 0.01, 6.0),
                Real("escapist_c", 0.01, 6.0),
                Real("rebel_c", 0.01, 6.0),
                Real("contrarian_c", 0.01, 6.0),
                Real("eschewer_c", 0.01, 6.0),
                Real("rejector_fraction", 0.01, 0.99),
                Real("defeatist_fraction", 0.01, 0.99),
                Real("escapist_fraction", 0.01, 0.99),
                Real("rebel_fraction", 0.01, 0.99),
                Real("contrarian_fraction", 0.01, 0.99),
                Real("eschewer_fraction", 0.01, 0.99),
                Bool("assign_roles_every_iteration"),
                *component_sparse_mask_params(),
            ],
        },

        'SparseHybridAdditivePSO': {
            'params': [
                *base_pso_params(),
                Real("rejector_c", 0.01, 6.0),
                Real("defeatist_c", 0.01, 6.0),
                Real("escapist_c", 0.01, 6.0),
                Real("rebel_c", 0.01, 6.0),
                Real("contrarian_c", 0.01, 6.0),
                Real("eschewer_c", 0.01, 6.0),
                Real("std_cognitive_prob", 0.01, 0.99),
                Real("rejector_prob", 0.01, 0.99),
                Real("defeatist_prob", 0.01, 0.99),
                Real("escapist_prob", 0.01, 0.99),
                Real("std_social_prob", 0.01, 0.99),
                Real("rebel_prob", 0.01, 0.99),
                Real("contrarian_prob", 0.01, 0.99),
                Real("eschewer_prob", 0.01, 0.99),
                Bool("assign_flags_every_iteration"),
                *component_sparse_mask_params(),
            ],
        },

    'AnarchicAmnesiacPSO': {
        'params': [
            Real("c1", 0.01, 6.0),
            Real("c2", 0.01, 6.0),
            Real("w", 0.01, 1.0),
            Real("anarchic_fraction", 0.01, 0.99),
            Real("amnesiac_fraction", 0.01, 0.99),
            Real("random_strength_social", 0.01, 6.0),
            Real("random_strength_cognitive", 0.01, 6.0),
        ],
    },

    'LSHADE': {
        'params': [
            # N_init = pop_size_factor * D (canonical r_init = 18, Tanabe &
            # Fukunaga 2014); dimension-relative, so it transfers to D=1000,
            # unlike an absolute initial_population_size.
            Integer("pop_size_factor", 2, 30),
            Integer("memory_size", 2, 50),
            Real("p_best_rate", 0.05, 0.25),
            Real("archive_size_rate", 1.0, 4.0),
        ]
    },

    'CMAES': {
        'params': [
            Integer("mu", 2, 100),
            Integer("lambda_", 10, 200)
        ],
        'forbidden': ["mu >= lambda_"]
    },

    'AAAPSO': {
        'params': [
            Real("c1", 0.01, 6.0),
            Real("c2", 0.01, 6.0),
            Real("base_inertia", 0.01, 1.0),
            Real("min_inertia", 0.01, 1.0),
            Real("max_inertia", 0.01, 1.0),
            Real("random_strength", 0.01, 6.0),
            Real("anarchic_fraction", 0.01, 0.8),
            Real("amnesiac_fraction", 0.01, 0.8),
            Integer("window_size", 5, 50),
            Real("max_anarchic_fraction", 0.01, 0.99),
            Real("max_amnesiac_fraction", 0.01, 0.99),
            Real("diversity_threshold", 0.001, 0.3, log=True),
            Real("improvement_threshold", 0.0001, 0.1, log=True),
        ],
    },

    'NAPSO': {
        'params': [
            Real("c1", 0.01, 6.0),
            Real("c2", 0.01, 6.0),
            Real("base_inertia", 0.01, 1.0),
            Real("min_inertia", 0.01, 1.0),
            Real("max_inertia", 0.01, 1.0),
            Real("noise_strength", 0.01, 3.0),
            Real("noisy_fraction", 0.01, 0.8),
            Real("max_noisy_fraction", 0.01, 0.99),
            Integer("window_size", 5, 50),
            Real("diversity_threshold", 0.001, 0.3, log=True),
            Real("improvement_threshold", 0.0001, 0.1, log=True),
        ],
    },

    'HybridFullDisjointPSO_WithRandom': {
        'params': [
            *base_pso_params(),
            Real("rejector_c", 0.01, 6.0),
            Real("defeatist_c", 0.01, 6.0),
            Real("escapist_c", 0.01, 6.0),
            Real("amnesiac_c", 0.01, 6.0),
            Real("rebel_c", 0.01, 6.0),
            Real("contrarian_c", 0.01, 6.0),
            Real("eschewer_c", 0.01, 6.0),
            Real("anarchic_c", 0.01, 6.0),
            Real("rejector_fraction", 0.01, 0.99),
            Real("defeatist_fraction", 0.01, 0.99),
            Real("escapist_fraction", 0.01, 0.99),
            Real("amnesiac_fraction", 0.01, 0.99),
            Real("rebel_fraction", 0.01, 0.99),
            Real("contrarian_fraction", 0.01, 0.99),
            Real("eschewer_fraction", 0.01, 0.99),
            Real("anarchic_fraction", 0.01, 0.99),
            Bool("assign_roles_every_iteration"),
        ],
    },

    'HybridPartialDisjointPSO_WithRandom': {
        'params': [
            *base_pso_params(),
            Real("rejector_c", 0.01, 6.0),
            Real("defeatist_c", 0.01, 6.0),
            Real("escapist_c", 0.01, 6.0),
            Real("amnesiac_c", 0.01, 6.0),
            Real("rebel_c", 0.01, 6.0),
            Real("contrarian_c", 0.01, 6.0),
            Real("eschewer_c", 0.01, 6.0),
            Real("anarchic_c", 0.01, 6.0),
            Real("rejector_fraction", 0.01, 0.99),
            Real("defeatist_fraction", 0.01, 0.99),
            Real("escapist_fraction", 0.01, 0.99),
            Real("amnesiac_fraction", 0.01, 0.99),
            Real("rebel_fraction", 0.01, 0.99),
            Real("contrarian_fraction", 0.01, 0.99),
            Real("eschewer_fraction", 0.01, 0.99),
            Real("anarchic_fraction", 0.01, 0.99),
            Bool("assign_roles_every_iteration"),
        ],
    },

    'HybridAdditivePSO_WithRandom': {
        'params': [
            *base_pso_params(),
            Real("rejector_c", 0.01, 6.0),
            Real("defeatist_c", 0.01, 6.0),
            Real("escapist_c", 0.01, 6.0),
            Real("amnesiac_c", 0.01, 6.0),
            Real("rebel_c", 0.01, 6.0),
            Real("contrarian_c", 0.01, 6.0),
            Real("eschewer_c", 0.01, 6.0),
            Real("anarchic_c", 0.01, 6.0),
            Real("std_cognitive_prob", 0.01, 0.99),
            Real("rejector_prob", 0.01, 0.99),
            Real("defeatist_prob", 0.01, 0.99),
            Real("escapist_prob", 0.01, 0.99),
            Real("amnesiac_prob", 0.01, 0.99),
            Real("std_social_prob", 0.01, 0.99),
            Real("rebel_prob", 0.01, 0.99),
            Real("contrarian_prob", 0.01, 0.99),
            Real("eschewer_prob", 0.01, 0.99),
            Real("anarchic_prob", 0.01, 0.99),
            Bool("assign_flags_every_iteration"),
        ],
    },

    'DrifterPSO': {
        'params': [
            *base_pso_params(),
            Real("drifter_fraction", 0.01, 0.99),
            Real("perturbation_scale", 0.0001, 0.1, log=True),
            Categorical("perturbation_method", ["gaussian", "cauchy"]),
        ],
    },

    'DAPSO': {
        'params': [
            Real("c1", 0.01, 6.0),
            Real("c2", 0.01, 6.0),
            Real("base_inertia", 0.01, 1.0),
            Real("min_inertia", 0.01, 1.0),
            Real("max_inertia", 0.01, 1.0),
            Real("perturbation_scale", 0.0001, 0.1, log=True),
            Categorical("perturbation_method", ["gaussian", "cauchy"]),
            Real("drifter_fraction", 0.01, 0.8),
            Real("max_drifter_fraction", 0.01, 0.99),
            Integer("window_size", 5, 50),
            Real("diversity_threshold", 0.001, 0.3, log=True),
            Real("improvement_threshold", 0.0001, 0.1, log=True),
        ],
    },

    'CLAPSO': {
        'params': [
            Real("c1", 0.01, 6.0),
            Real("c2", 0.01, 6.0),
            Real("cl_c1", 0.01, 6.0),
            Real("cl_c2", 0.01, 6.0),
            Real("b1", 0.01, 6.0),
            Real("b2", 0.01, 6.0),
            Real("base_inertia", 0.01, 1.0),
            Real("min_inertia", 0.01, 1.0),
            Real("max_inertia", 0.01, 1.0),
            Real("cl_fraction", 0.01, 0.8),
            Real("max_cl_fraction", 0.01, 0.99),
            Integer("window_size", 5, 50),
            Real("diversity_threshold", 0.001, 0.3, log=True),
            Real("improvement_threshold", 0.0001, 0.1, log=True),
        ],
    },

    # ----- baselines -----

    'PSO': {
        'params': [
            *base_pso_params(),
        ],
    },

    'PerturbationPSO': {
        'params': [
            *base_pso_params(),
            Real("perturbation_scale", 0.0001, 0.1, log=True),
            Categorical("perturbation_method", ["gaussian", "cauchy"]),
        ],
    },

    'GeneticAlgorithm': {
        'params': [
            Real("crossover_probability", 0.6, 1.0),
            Real("sbx_distribution_index", 2.0, 30.0, log=True),
            Real("mutation_distribution_index", 5.0, 100.0, log=True),
        ],
    },

    'DifferentialEvolution': {
        'params': [
            Real("CR", 0.0, 1.0),
            Real("F", 0.1, 1.0),
        ],
    },

    # ----- single-role (non-sparse) -----

    'RebelPSO': {
        'params': [
            *base_pso_params(),
            Real("ac2", 0.01, 6.0),
            Real("rebel_fraction", 0.01, 0.99),
        ],
    },

    'RejectorPSO': {
        'params': [
            *base_pso_params(),
            Real("ac1", 0.01, 6.0),
            Real("rejector_fraction", 0.01, 0.99),
        ],
    },

    'ContrarianPSO': {
        'params': [
            *base_pso_params(),
            Real("ac2", 0.01, 6.0),
            Real("contrarian_fraction", 0.01, 0.99),
        ],
    },

    'DefeatistPSO': {
        'params': [
            *base_pso_params(),
            Real("ac1", 0.01, 6.0),
            Real("defeatist_fraction", 0.01, 0.99),
        ],
    },

    'EschewerPSO': {
        'params': [
            *base_pso_params(),
            Real("ac2", 0.01, 6.0),
            Real("eschewer_fraction", 0.01, 0.99),
        ],
    },

    'EscapistPSO': {
        'params': [
            *base_pso_params(),
            Real("ac1", 0.01, 6.0),
            Real("escapist_fraction", 0.01, 0.99),
        ],
    },

    'AnarchicPSO': {
        'params': [
            *base_pso_params(),
            Real("random_strength", 0.01, 6.0),
            Real("anarchic_fraction", 0.01, 0.99),
        ],
    },

    'AmnesiacPSO': {
        'params': [
            *base_pso_params(),
            Real("random_strength", 0.01, 6.0),
            Real("amnesiac_fraction", 0.01, 0.99),
        ],
    },

    'ErraticPSO': {
        'params': [
            *base_pso_params(),
            Real("random_strength", 0.01, 6.0),
            Real("erratic_fraction", 0.01, 0.99),
        ],
    },

    'WandererPSO': {
        'params': [
            *base_pso_params(),
            Real("noise_strength", 0.01, 3.0),
            Real("wanderer_fraction", 0.01, 0.99),
        ],
    },

    # ----- two-role (non-sparse) -----

    'RebelRejectorPSO': {
        'params': [
            *base_pso_params(),
            Real("ac1", 0.01, 6.0),
            Real("ac2", 0.01, 6.0),
            Real("rebel_fraction", 0.01, 0.99),
            Real("rejector_fraction", 0.01, 0.99),
        ],
    },

    'ContrarianDefeatistPSO': {
        'params': [
            *base_pso_params(),
            Real("ac1", 0.01, 6.0),
            Real("ac2", 0.01, 6.0),
            Real("contrarian_fraction", 0.01, 0.99),
            Real("defeatist_fraction", 0.01, 0.99),
        ],
    },

    'EschewerEscapistPSO': {
        'params': [
            *base_pso_params(),
            Real("ac1", 0.01, 6.0),
            Real("ac2", 0.01, 6.0),
            Real("eschewer_fraction", 0.01, 0.99),
            Real("escapist_fraction", 0.01, 0.99),
        ],
    },

    # ----- adaptive two-role -----

    'RRAPSO': {
        'params': [
            Real("c1", 0.01, 6.0),
            Real("c2", 0.01, 6.0),
            Real("ac1", 0.01, 6.0),
            Real("ac2", 0.01, 6.0),
            Real("base_inertia", 0.01, 1.0),
            Real("min_inertia", 0.01, 1.0),
            Real("max_inertia", 0.01, 1.0),
            Real("rebel_fraction", 0.01, 0.8),
            Real("rejector_fraction", 0.01, 0.8),
            Real("max_rebel_fraction", 0.01, 0.99),
            Real("max_rejector_fraction", 0.01, 0.99),
            Integer("window_size", 5, 50),
            Real("diversity_threshold", 0.001, 0.3, log=True),
            Real("improvement_threshold", 0.0001, 0.1, log=True),
        ],
    },

    'CDAPSO': {
        'params': [
            Real("c1", 0.01, 6.0),
            Real("c2", 0.01, 6.0),
            Real("ac1", 0.01, 6.0),
            Real("ac2", 0.01, 6.0),
            Real("base_inertia", 0.01, 1.0),
            Real("min_inertia", 0.01, 1.0),
            Real("max_inertia", 0.01, 1.0),
            Real("contrarian_fraction", 0.01, 0.8),
            Real("defeatist_fraction", 0.01, 0.8),
            Real("max_contrarian_fraction", 0.01, 0.99),
            Real("max_defeatist_fraction", 0.01, 0.99),
            Integer("window_size", 5, 50),
            Real("diversity_threshold", 0.001, 0.3, log=True),
            Real("improvement_threshold", 0.0001, 0.1, log=True),
        ],
    },

    'EEAPSO': {
        'params': [
            Real("c1", 0.01, 6.0),
            Real("c2", 0.01, 6.0),
            Real("ac1", 0.01, 6.0),
            Real("ac2", 0.01, 6.0),
            Real("base_inertia", 0.01, 1.0),
            Real("min_inertia", 0.01, 1.0),
            Real("max_inertia", 0.01, 1.0),
            Real("eschewer_fraction", 0.01, 0.8),
            Real("escapist_fraction", 0.01, 0.8),
            Real("max_eschewer_fraction", 0.01, 0.99),
            Real("max_escapist_fraction", 0.01, 0.99),
            Integer("window_size", 5, 50),
            Real("diversity_threshold", 0.001, 0.3, log=True),
            Real("improvement_threshold", 0.0001, 0.1, log=True),
        ],
    },

    # ----- reverse learning -----

    'ReverseLearningPSO': {
        'params': [
            Real("w", 0.01, 1.0),
            Real("b1", 0.01, 6.0),
            Real("b2", 0.01, 6.0),
        ],
    },

    'ReverseLearningGlobalAttractorPSO': {
        'params': [
            Real("w", 0.01, 1.0),
            Real("a", 0.01, 6.0),
            Real("b1", 0.01, 6.0),
            Real("b2", 0.01, 6.0),
        ],
    },

    'ReverseLearningPersonalAttractorPSO': {
        'params': [
            Real("w", 0.01, 1.0),
            Real("a", 0.01, 6.0),
            Real("b1", 0.01, 6.0),
            Real("b2", 0.01, 6.0),
        ],
    },

    'CombinedLearningPSO': {
        'params': [
            *base_pso_params(),
            Real("b1", 0.01, 6.0),
            Real("b2", 0.01, 6.0),
        ],
    },

    # ----- co-/individually-adaptive coefficients -----

    'CAPSO': {
        'params': [
            *base_pso_params(),
            Real("max_c1", 0.01, 6.0),
            Real("max_c2", 0.01, 6.0),
        ],
    },

    'IAPSO': {
        'params': [
            *base_pso_params(),
            Real("max_c1", 0.01, 6.0),
            Real("max_c2", 0.01, 6.0),
        ],
    },

    # ----- base role hybrids (six deliberative roles, no random roles) -----

    'HybridFullDisjointPSO': {
        'params': [
            *base_pso_params(),
            Real("rejector_c", 0.01, 6.0),
            Real("defeatist_c", 0.01, 6.0),
            Real("escapist_c", 0.01, 6.0),
            Real("rebel_c", 0.01, 6.0),
            Real("contrarian_c", 0.01, 6.0),
            Real("eschewer_c", 0.01, 6.0),
            Real("rejector_fraction", 0.01, 0.99),
            Real("defeatist_fraction", 0.01, 0.99),
            Real("escapist_fraction", 0.01, 0.99),
            Real("rebel_fraction", 0.01, 0.99),
            Real("contrarian_fraction", 0.01, 0.99),
            Real("eschewer_fraction", 0.01, 0.99),
            Bool("assign_roles_every_iteration"),
        ],
    },

    'HybridPartialDisjointPSO': {
        'params': [
            *base_pso_params(),
            Real("rejector_c", 0.01, 6.0),
            Real("defeatist_c", 0.01, 6.0),
            Real("escapist_c", 0.01, 6.0),
            Real("rebel_c", 0.01, 6.0),
            Real("contrarian_c", 0.01, 6.0),
            Real("eschewer_c", 0.01, 6.0),
            Real("rejector_fraction", 0.01, 0.99),
            Real("defeatist_fraction", 0.01, 0.99),
            Real("escapist_fraction", 0.01, 0.99),
            Real("rebel_fraction", 0.01, 0.99),
            Real("contrarian_fraction", 0.01, 0.99),
            Real("eschewer_fraction", 0.01, 0.99),
            Bool("assign_roles_every_iteration"),
        ],
    },

    'HybridAdditivePSO': {
        'params': [
            *base_pso_params(),
            Real("rejector_c", 0.01, 6.0),
            Real("defeatist_c", 0.01, 6.0),
            Real("escapist_c", 0.01, 6.0),
            Real("rebel_c", 0.01, 6.0),
            Real("contrarian_c", 0.01, 6.0),
            Real("eschewer_c", 0.01, 6.0),
            Real("std_cognitive_prob", 0.01, 0.99),
            Real("rejector_prob", 0.01, 0.99),
            Real("defeatist_prob", 0.01, 0.99),
            Real("escapist_prob", 0.01, 0.99),
            Real("std_social_prob", 0.01, 0.99),
            Real("rebel_prob", 0.01, 0.99),
            Real("contrarian_prob", 0.01, 0.99),
            Real("eschewer_prob", 0.01, 0.99),
            Bool("assign_flags_every_iteration"),
        ],
    },

    # ----- wanderer hybrids -----

    'HybridDisjointPSO_WithWanderer': {
        'params': [
            *base_pso_params(),
            Real("wanderer_c", 0.01, 6.0),
            Real("wanderer_fraction", 0.01, 0.99),
            Bool("assign_roles_every_iteration"),
        ],
    },

    'HybridAdditivePSO_WithWanderer': {
        'params': [
            *base_pso_params(),
            Real("wanderer_c", 0.01, 6.0),
            Real("std_cognitive_prob", 0.01, 0.99),
            Real("std_social_prob", 0.01, 0.99),
            Real("wanderer_prob", 0.01, 0.99),
            Bool("assign_flags_every_iteration"),
        ],
    },

    # ----- PSO-GA hybrids (SBX/mutation built in the target runner) -----

    'PGSHEA': {
        'params': [
            *base_pso_params(),
            Integer("swap_interval", 1, 100, log=True),
            Categorical("starting_algorithm", ["PSO", "GA"]),
            Real("crossover_probability", 0.6, 1.0),
            Real("sbx_distribution_index", 2.0, 30.0, log=True),
            Real("mutation_distribution_index", 5.0, 100.0, log=True),
        ],
    },

    'PGPHEA': {
        'params': [
            *base_pso_params(),
            Integer("exchange_interval", 1, 100, log=True),
            Integer("exchange_number", 1, 50),
            Real("crossover_probability", 0.6, 1.0),
            Real("sbx_distribution_index", 2.0, 30.0, log=True),
            Real("mutation_distribution_index", 5.0, 100.0, log=True),
        ],
    },
}

current_algorithm = None


# ==============================================================================
# Constraint Handling Functions
# ==============================================================================
def normalize_fraction_sum(fractions_dict: dict, max_sum: float = 1.0) -> dict:
    """Normalizes fractions in a dictionary if their sum exceeds max_sum."""
    current_sum = sum(v for v in fractions_dict.values() if isinstance(v, (int, float)))
    numeric_fractions = {k:v for k,v in fractions_dict.items() if isinstance(v, (int, float))}
    normalized_fractions = numeric_fractions.copy() # Work with numeric copy

    if current_sum > max_sum + 1e-9:
        print(f"Normalizing fractions (Sum: {current_sum:.4f} > {max_sum})")
        if current_sum > 1e-9:
            factor = max_sum / current_sum
            for role in normalized_fractions:
                normalized_fractions[role] *= factor
        else:
            for role in normalized_fractions: normalized_fractions[role] = 0.0
    return normalized_fractions


def repair_max_param_constraints_random(config: dict) -> dict:
    """
    Repairs constraints like max_param >= param and min <= base <= max.
    Uses swapping for invalid min/max ranges, random repair for base, clamping for simple max >= param.
    """
    repaired_config = config.copy()
    # Max >= Param constraints (clamp max = param)
    constraints_to_check = [
        ("c1", "max_c1"), ("c2", "max_c2"),
        ("eschewer_fraction", "max_eschewer_fraction"), ("escapist_fraction", "max_escapist_fraction"),
        ("contrarian_fraction", "max_contrarian_fraction"), ("defeatist_fraction", "max_defeatist_fraction"),
        ("rebel_fraction", "max_rebel_fraction"), ("rejector_fraction", "max_rejector_fraction"),
        ("anarchic_fraction", "max_anarchic_fraction"), ("amnesiac_fraction", "max_amnesiac_fraction"),
        ("noisy_fraction", "max_noisy_fraction"), ("cl_fraction", "max_cl_fraction"),
        ("drifter_fraction", "max_drifter_fraction")
    ]
    for param, max_param in constraints_to_check:
        if param in repaired_config and max_param in repaired_config:
            param_val = repaired_config[param]; max_param_val = repaired_config[max_param]
            if isinstance(param_val, (int, float)) and isinstance(max_param_val, (int, float)):
                if max_param_val < param_val:
                    print(f"Repair constraint: {max_param}<{param}. Clamp {max_param}={param_val:.4f}")
                    repaired_config[max_param] = param_val

    # Min <= Base <= Max constraints (swap min/max, random repair base)
    min_base_max_triplets = [("min_inertia", "base_inertia", "max_inertia")]
    for min_key, base_key, max_key in min_base_max_triplets:
        if min_key in repaired_config and base_key in repaired_config and max_key in repaired_config:
            min_val = repaired_config[min_key]; base_val = repaired_config[base_key]; max_val = repaired_config[max_key]
            if not (isinstance(min_val, (int, float)) and isinstance(base_val, (int, float)) and isinstance(max_val, (int, float))): continue
            if max_val < min_val:
                print(f"Repair bounds: {max_key}<{min_key}. Swap.")
                repaired_config[min_key], repaired_config[max_key] = max_val, min_val
                min_val, max_val = max_val, min_val # Update local vars
            # Step 2: Repair base_val if outside [min_val, max_val]
            if base_val < min_val or base_val > max_val:
                 if max_val >= min_val:
                      new_base = random.uniform(min_val, max_val)
                      print(f"Repair value: {base_key} ({base_val:.4f}) outside [{min_val:.4f}, {max_val:.4f}]. Set to random: {new_base:.4f}")
                      repaired_config[base_key] = new_base
                 else: # Should not happen after swap, but safety check
                      print(f"Warning: Invalid range [{min_val:.4f}, {max_val:.4f}] after repair for {base_key}. Clamping to min.")
                      repaired_config[base_key] = min_val

    return repaired_config

# ==============================================================================
# Main Target Runner Function
# ==============================================================================

COGNITIVE_FRACTION_KEYS = ["rejector_fraction", "defeatist_fraction", "escapist_fraction", "amnesiac_fraction"]
SOCIAL_FRACTION_KEYS = ["rebel_fraction", "contrarian_fraction", "eschewer_fraction", "anarchic_fraction"]
ALL_SPECIAL_FRACTION_KEYS = COGNITIVE_FRACTION_KEYS + SOCIAL_FRACTION_KEYS + ["wanderer_fraction"]

PARTIAL_DISJOINT_ALGORITHMS = {
    'HybridPartialDisjointPSO',
    'HybridPartialDisjointPSO_WithRandom',
    'HybridPartialDisjointRestarterPSO',
    'SparseHybridPartialDisjointPSO',
}
FULL_DISJOINT_ALGORITHMS = {
    'HybridFullDisjointPSO',
    'HybridFullDisjointPSO_WithRandom',
    'HybridFullDisjointRestarterPSO',
    'HybridDisjointPSO_WithWanderer',
    'SparseHybridFullDisjointPSO',
}


def repair_and_normalize_config(algo_name: str, config: dict) -> dict:
    """Repair max/min constraints and apply the algorithm family's fraction
    normalization. This is the exact transformation the target runner applies
    before constructing an algorithm; applying it to the saved winners makes
    them ready to transcribe into experiment/factories.py (raw irace output
    can violate the constructors' fraction constraints)."""
    repaired = repair_max_param_constraints_random(config)
    final_params = repaired.copy()

    # Sparse coordinate masks: the mode is fixed to "fraction" (a fraction of
    # coordinates transfers unchanged across dimensions, unlike sqrt/log/count).
    # Injected here so both the evaluated config and the saved winners carry it
    # (the constructors default to "sqrt").
    for mode_key, fraction_key in (
        ("coordinate_mode", "coordinate_fraction"),
        ("social_coordinate_mode", "social_coordinate_fraction"),
        ("cognitive_coordinate_mode", "cognitive_coordinate_fraction"),
    ):
        if fraction_key in final_params:
            final_params.setdefault(mode_key, "fraction")

    if algo_name in PARTIAL_DISJOINT_ALGORITHMS:
        for keys in (COGNITIVE_FRACTION_KEYS, SOCIAL_FRACTION_KEYS):
            group = {k: repaired.get(k, 0.0) for k in keys if k in repaired}
            if group:
                final_params.update(normalize_fraction_sum(group, 1.0))
    elif algo_name in FULL_DISJOINT_ALGORITHMS:
        group = {k: repaired.get(k, 0.0) for k in ALL_SPECIAL_FRACTION_KEYS if k in repaired}
        if group:
            final_params.update(normalize_fraction_sum(group, 1.0))

    return final_params


def derive_run_seed(base_seed: int, problem_name: str, run_index: int) -> int:
    identity = f"{base_seed}:{problem_name}:{run_index}"
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def target_runner(experiment: Experiment, scenario: Scenario) -> float:
    """
    Universal target runner. Repairs max/min constraints.
    Performs fraction normalization *within the runner* based on algorithm type.
    """
    global current_algorithm, problems, num_runs, solutions_size, max_evaluations # Access globals

    if current_algorithm is None: return float('inf')
    config = experiment.configuration

    # Steps 1+2: repair max/min constraints and apply the algorithm family's
    # fraction normalization (shared with the save path, so stored winners
    # are exactly what was evaluated).
    final_params = repair_and_normalize_config(current_algorithm, config)

    # --- Special handling: algorithms needing operator objects built from
    # scalar tuned parameters (the scalars are what gets saved/transcribed) ---
    if current_algorithm in ('PGCHEA', 'PGSHEA', 'PGPHEA'):
        final_params['solutions_size'] = solutions_size
        final_params['crossover'] = SBXCrossover(
            probability=final_params.pop('crossover_probability', 1.0),
            distribution_index=final_params.pop('sbx_distribution_index', 5.0),
        )
        final_params['mutation'] = PolynomialMutation(
            probability=1.0 / number_of_variables,
            distribution_index=final_params.pop('mutation_distribution_index', 20.0),
        )
    elif current_algorithm == 'GeneticAlgorithm':
        final_params['population_size'] = solutions_size
        final_params['offspring_population_size'] = solutions_size
        final_params['crossover'] = SBXCrossover(
            probability=final_params.pop('crossover_probability', 0.9),
            distribution_index=final_params.pop('sbx_distribution_index', 20.0),
        )
        final_params['mutation'] = PolynomialMutation(
            probability=1.0 / number_of_variables,
            distribution_index=final_params.pop('mutation_distribution_index', 20.0),
        )
    elif current_algorithm == 'DifferentialEvolution':
        de_f = final_params.pop('F', 0.5)
        final_params['crossover_operator'] = DifferentialEvolutionCrossover(
            CR=final_params.pop('CR', 0.9), F=de_f, K=de_f,
        )

    # --- Step 3: Get Algorithm Class ---
    try:
        AlgorithmClass = globals()[current_algorithm]
    except KeyError:
        print(f"ERROR: Algorithm class '{current_algorithm}' not found.")
        return float('inf')

    # --- Step 4: Run Experiments (one irace instance = one problem) ---
    # irace races configurations ACROSS instances with rank-based elimination,
    # so the wildly different cost scales of the tuning problems never mix
    # (raw-mean aggregation let Rastrigin's ~600 dominate Ackley's ~13).
    results = []
    tuning_problems = [experiment.instance] if experiment.instance is not None else problems
    problem_obj_for_run = None # To hold the problem instance for this set of runs
    for problem in tuning_problems:
        problem_name = problem.get_name() if hasattr(problem, 'get_name') else problem.__class__.__name__
        problem_obj_for_run = problem # Use the selected problem
        # print(f"  Problem: {problem_name}") # Verbose
        for run_index in range(num_runs):
            try:
                # Seed the algorithm RNGs for this run (irace supplies a
                # per-experiment seed; runs and problems get distinct streams).
                run_seed = derive_run_seed(experiment.seed, problem_name, run_index)
                random.seed(run_seed)
                np.random.seed(run_seed)
                if hasattr(problem_obj_for_run, "set_seed"):
                    problem_obj_for_run.set_seed(run_seed)

                # Prepare parameters, FILTERING based on the specific AlgorithmClass constructor
                constructor_params = {"problem": problem_obj_for_run, # Use the problem instance
                                      "swarm_size": solutions_size,
                                      "termination_criterion": StoppingByEvaluations(max_evaluations)}
                constructor_params.update(final_params) # Add repaired/normalized params

                sig = inspect.signature(AlgorithmClass.__init__)
                allowed_params = {k for k in sig.parameters if k != 'self'}
                use_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

                if use_kwargs:
                    filtered_constructor_params = constructor_params # Pass all if **kwargs allowed
                else:
                    filtered_constructor_params = {k: v for k, v in constructor_params.items() if k in allowed_params}

                # Instantiate
                algorithm = AlgorithmClass(**filtered_constructor_params)
                algorithm.run()
                result = algorithm.result()

                if result is None or not hasattr(result, 'objectives') or not result.objectives:
                     results.append(float('inf'))
                else:
                    results.append(result.objectives[0])

            except Exception as e:
                print(f"      ERROR during run {run_index + 1} on {problem_name} for {current_algorithm}: {e}")
                # Print config for debugging, NOT filtered params as they vary per algo
                print(f"      Config (Original): {config}")
                print(f"      Config (Repaired+Normalized): {final_params}")
                print(f"      Final Params Passed (Subset): {{k:v for k,v in filtered_constructor_params.items() if k not in ['problem','termination_criterion']}}") # Show relevant params
                traceback.print_exc()
                results.append(float('inf'))

    # --- Step 5: Process Results ---
    # print(f"--- Finished runs for config. Num results: {len(results)} ---")
    if not results: return float('inf')
    try:
        numeric_results = [r for r in results if isinstance(r, (int, float)) and np.isfinite(r)]
        if not numeric_results: return float('inf')
        avg_result = np.mean(numeric_results)
    except Exception as e: print(f"Error calculating mean: {e}"); return float('inf')
    cost = float(avg_result) if np.isfinite(avg_result) else float('inf')
    # Log original config for irace, but final cost
    print(f"Evaluated config ({final_params}) -> Final Avg Cost: {cost:.4f}")
    return cost


def _load_configurations(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@contextmanager
def _file_lock(path: str):
    """Short-lived exclusive lock (O_CREAT|O_EXCL lock file) serializing
    read-merge-write cycles between concurrent jobs on a shared filesystem."""
    lock_path = path + ".lock"
    for _ in range(600):  # wait up to ~60 s for the lock
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            time.sleep(0.1)
    else:
        raise TimeoutError(
            f"Could not acquire {lock_path}. If a previous job crashed while "
            f"saving, delete the stale lock file and rerun.")
    try:
        yield
    finally:
        os.close(lock_fd)
        os.remove(lock_path)


def _save_configurations(path: str, algo_name: str, records: list) -> dict:
    """Merge one algorithm's results into the shared JSON, safely for
    concurrent SLURM jobs: the lock serializes the read-merge-write cycle,
    and the write is an atomic rename, so other jobs never see a half-written
    file and never lose each other's entries."""
    with _file_lock(path):
        merged = _load_configurations(path)
        merged[algo_name] = records
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(merged, f, indent=4,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))
        os.replace(tmp_path, path)
    return merged


CLAIM_HEARTBEAT_SECONDS = 60
CLAIM_STALE_SECONDS = 600  # 10 missed heartbeats: the claiming job is dead


class _Claim:
    """A live "in progress" marker for one algorithm.

    A daemon heartbeat thread touches the claim file once a minute; it dies
    with the process (crash, OOM, scancel, walltime), so the file's mtime
    freezes and other jobs take the block over after CLAIM_STALE_SECONDS.
    """

    def __init__(self, path: str):
        self.path = path
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()

    def _beat(self):
        while not self._stop.wait(CLAIM_HEARTBEAT_SECONDS):
            try:
                os.utime(self.path)
            except OSError:
                pass  # claim gone or FS hiccup; staleness handling covers it

    def release(self):
        self._stop.set()
        self._thread.join(timeout=5)
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass


def _try_claim(output_file: str, algo_name: str):
    """Atomically claim one algorithm block. Returns a live _Claim, or None
    when the block is already finished or freshly claimed by another job.
    Stale claims (no heartbeat for CLAIM_STALE_SECONDS) are taken over."""
    claim_dir = os.path.join(os.path.dirname(output_file), "irace_claims")
    os.makedirs(claim_dir, exist_ok=True)
    claim_path = os.path.join(claim_dir, f"{algo_name}.claim")
    with _file_lock(output_file):
        if algo_name in _load_configurations(output_file):
            return None  # finished by another job while we were deciding
        if os.path.exists(claim_path):
            age = time.time() - os.path.getmtime(claim_path)
            if age < CLAIM_STALE_SECONDS:
                return None  # another job is actively tuning this block
            print(f"Taking over stale claim for {algo_name} "
                  f"(no heartbeat for {age:.0f} s).")
        with open(claim_path, "w") as f:
            json.dump({
                "job": os.environ.get("SLURM_JOB_ID", "local"),
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "claimed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=4)
    return _Claim(claim_path)


if __name__ == "__main__":
    output_file = os.path.join("optimization_results", "irace_best_configurations.json")
    os.makedirs("optimization_results", exist_ok=True)

    for algo_name, space_config in parameter_spaces.items():
        # Resume support: results are saved after every algorithm, so a
        # restarted job skips finished blocks. Re-read before each block so
        # concurrently running jobs also see each other's finished algorithms.
        # NOTE: entries from an older tuning protocol also count as "done" -
        # archive/delete the JSON before starting a fresh campaign.
        best_configurations = _load_configurations(output_file)
        if algo_name in best_configurations:
            print(f"SKIPPING {algo_name}: {len(best_configurations[algo_name])} "
                  f"configuration(s) already stored in {output_file}.")
            continue

        # Claim the block so concurrent jobs pick different algorithms.
        claim = _try_claim(output_file, algo_name)
        if claim is None:
            print(f"SKIPPING {algo_name}: being tuned by another job "
                  f"(or finished just now).")
            continue

        current_algorithm = algo_name
        print(f"Optimizing parameters for {algo_name} ...")

        try:
            # Unpack the parameters list and the forbidden expression from the config
            params_list = space_config['params']
            forbidden_expression = space_config.get('forbidden', None)

            # Create the ParameterSpace using the extracted components
            parameter_space = ParameterSpace(params=params_list, forbidden=forbidden_expression)

            # Parallel workers: match the SLURM allocation (48 on Ares nodes,
            # 28 on Eagle); falls back to 48 for manual runs.
            n_jobs = int(os.environ.get("SLURM_CPUS_ON_NODE", 48))
            scenario = Scenario(max_experiments=budget * len(params_list), instances=problems, seed=42, n_jobs=n_jobs)

            result = irace(target_runner, parameter_space, scenario, return_df=True, remove_metadata=True)
            # Store READY-TO-TRANSCRIBE configurations: the same repair + fraction
            # normalization the target runner applied during evaluation.
            tuned_records = [
                repair_and_normalize_config(algo_name, dict(row))
                for row in result.to_dict(orient="records")
            ]

            # 2. Load that RData into R’s global env (it creates `iraceResults`)
            # robjects.r['load']("irace.log")

            # 3. Tell IRACE to dump the human‑readable log to a .txt file
            # robjects.r['save_irace_logfile'](robjects.r['iraceResults'], "irace.txt")

            # Save results after each algorithm (lock + re-read + merge + atomic
            # rename: concurrent jobs cannot clobber each other's entries).
            best_configurations = _save_configurations(output_file, algo_name, tuned_records)
        finally:
            # Always release the claim: on success the JSON entry takes
            # over; on failure another job may retry the block.
            claim.release()
