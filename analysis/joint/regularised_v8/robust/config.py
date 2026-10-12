"""Fixed scientific choices. No held-out-data-driven numerical or penalty tuning."""
from dataclasses import dataclass,asdict

@dataclass(frozen=True)
class Config:
    C: float=0.02
    h0_ridge: float=1e-3
    alpha_floor: float=1e-8
    requested_intervals: int=32
    quadrature_points: int=5
    interval_min_gap: float=1e-8
    head_tolerance: float=1e-5
    alpha_gap_tolerance: float=1e-5
    joint_max_iterations: int=1000
    max_working_rounds: int=30
    working_batch: int=64
    max_working_columns: int=512
    zero_tolerance: float=1e-10
    objective_acceptance_slack: float=1e-11
    rescue_cycles: int=30
    head_max_iterations: int=3000
    alpha_max_iterations: int=500
    max_fit_seconds: float=1800.
    repeats: int=5
    outer_folds: int=5
    def payload(self): return asdict(self)
    def validate(self):
        if self.C<=0 or self.h0_ridge<=0 or not 0<self.alpha_floor<1:
            raise ValueError('C and the explicitly declared H0 ridge must be positive.')
        if min(self.head_tolerance,self.alpha_gap_tolerance,self.max_fit_seconds)<=0:
            raise ValueError('Invalid numerical configuration.')
        if min(self.max_working_rounds,self.working_batch,self.joint_max_iterations)<1:
            raise ValueError('Invalid iteration settings.')

SCHEMA='joint-ibd-regularised-v8.0.0'
TASK='IBD_vs_nonIBD'
CLASSES=('nonIBD','IBD')

def specification(config):
    return {'schema':SCHEMA,'task':TASK,'classes':list(CLASSES),'configuration':config.payload(),
      'objective':'class_balanced_mean_CE + ||beta_R||_1/(C*sum_w) + h0_ridge/2*||beta_H||_2^2',
      'alpha_weighting':'source_interval_before_original_death_axis_integration',
      'alpha_penalty':0.,'h0_L1_penalty':0.,'intercept_penalty':0.,
      'resolution_selection':False,'inner_fits':0,'expected_pairs':25,
      'repetitions':[1,2,3,4,5],'fresh_fit_initialisation':'alpha one and zero head',
      'original_ridge_free_model':False,'global_optimum_claimed':False,
      'stopping':'full head KKT and conditional alpha FW gap, not the old 100-refit stability rule',
      'finite_budget_policy':'retain the best finite training-objective state; report approximation explicitly if tolerances not reached',
      'score_partial_experiment_as_complete':False,'unsafe_or_nonfinite_models_scored':False,
      'controls':'matched Ricci-only model with same C, sample weights, scaling and numerical head checks',
      'historical_outputs_overwritten':False,
      'upstream':'Existing source cohort graph scaffold, references and axis are not reconstructed per fold.',
      'interpretation':'Exploratory feasible regularised complementarity analysis, not the original unpenalised nested experiment.'}
