"""One scientific configuration. Production CLI does not expose penalty/tolerance tuning."""
from dataclasses import dataclass, asdict

TASK_ORDER = ('IBD_vs_nonIBD','three_way_nonIBD_UC_CD','nonIBD_vs_UC','nonIBD_vs_CD','CD_vs_UC')
MATRIX_SHA256 = '4eca6a4cdc2d6f408be9f3f8428a43bf5fe1d56c599e06a198c57827017b006b'

@dataclass(frozen=True)
class Settings:
    repeats: int = 20
    outer_folds: int = 5
    inner_folds: int = 3
    interval_grid: tuple = (32,64,96,128,160)
    C: float = 0.02
    lambda_h0: float = 0.0
    lambda_ricci: float = 1.0
    alpha_smoothness: float = 0.0
    alpha_floor: float = 1e-8
    quadrature_points: int = 5
    min_interval_width: float = 1e-8
    head_tolerance: float = 1e-6
    alpha_gap_tolerance: float = 1e-6
    min_alternations: int = 2
    max_alternations: int = 500
    alpha_slsqp_iterations: int = 1000
    alpha_pairwise_iterations: int = 3000
    acceleration_every: int = 5
    acceleration_iterations: int = 150
    head_seconds: float = 600.0
    fit_seconds: float = 3600.0
    separation_tolerance: float = 1e-7
    selection_metric: str = 'mean_inner_sample_roc_auc'
    seed_policy: str = 'original_outer_seed_plus_100_times_outer_fold_for_inner'
    def payload(self): return asdict(self)
    def validate(self):
        if self.C<=0 or self.lambda_h0!=0 or self.lambda_ricci!=1 or self.alpha_smoothness!=0:
            raise ValueError('This runner is specifically unpenalised H0, fixed Ricci; configuration changed.')
        if not 0<self.alpha_floor<1: raise ValueError('Invalid alpha positivity floor.')
        if len(set(self.interval_grid))!=len(self.interval_grid) or min(self.interval_grid)<2:
            raise ValueError('Invalid interval grid.')
