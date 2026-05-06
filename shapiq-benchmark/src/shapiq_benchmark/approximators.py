from __future__ import annotations
import os
from pathlib import Path


import numpy as np
from shapiq.approximator import SVARMIQ, SHAPIQ, KernelSHAPIQ, ProxySPEX
from proxyshap.proxyshap import ProxySHAP, ProxySHAPHPO
from shapiq.approximator.permutation.sii import PermutationSamplingSII
from shapiq.approximator.permutation.sv import PermutationSamplingSV
from shapiq.approximator.sparse.proxyspex import (
    ProxySPEXXGBoost,
    ProxySPEXXGBoostNoRefinement,
    ProxySPEXXGBoostNoTruncationNoRefinement,
    ProxySPEXLightGBM,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def get_approximators(
    APPROXIMATORS, NPLAYERS, RANDOMSTATE, PAIRING, INDEX, MAXORDER, n_estimators=None
):
    approximator_list = []
    sampling_weights = np.ones(NPLAYERS + 1)
    proxy_model = DecisionTreeRegressor(random_state=RANDOMSTATE, max_depth=6)
    if "SVARM" in APPROXIMATORS or "SVARMIQ" in APPROXIMATORS:
        svarmiq = SVARMIQ(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
        )
        svarmiq.name = "SVARMIQ"
        approximator_list.append(svarmiq)
    if "SHAPIQ" in APPROXIMATORS:
        shapiq = SHAPIQ(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
        )
        shapiq.name = "SHAPIQ"
        approximator_list.append(shapiq)
    if "ProxySPEX" in APPROXIMATORS:
        proxy_spex = ProxySPEX(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            index=INDEX,
            max_order=MAXORDER,
        )
        proxy_spex.name = "ProxySPEX"
        approximator_list.append(proxy_spex)
    if "KernelSHAPIQ" in APPROXIMATORS:
        kernel_shapiq = KernelSHAPIQ(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
        )
        kernel_shapiq.name = "KernelSHAPIQ"
        approximator_list.append(kernel_shapiq)
    if "PermutationSamplingSII" in APPROXIMATORS or "PermutationSamplingSV" in APPROXIMATORS:
        if INDEX == "SV":
            permutation_sampling = PermutationSamplingSV(
                n=NPLAYERS, pairing_trick=PAIRING, random_state=RANDOMSTATE
            )
            permutation_sampling.name = "PermutationSamplingSV"
            approximator_list.append(permutation_sampling)
        else:
            # Permutation Sampling
            permutation_sampling = PermutationSamplingSII(
                n=NPLAYERS, max_order=MAXORDER, index=INDEX, random_state=RANDOMSTATE
            )
            permutation_sampling.name = "PermutationSamplingSII"
            approximator_list.append(permutation_sampling)
    if "ProxySHAP (XGBoost)" in APPROXIMATORS:
        proxyshap_no_adjust = ProxySHAP(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            adjustment="none",
            sampling_weights=sampling_weights,
        )
        proxyshap_no_adjust.name = "ProxySHAP (XGBoost)"
        approximator_list.append(proxyshap_no_adjust)
    if "ProxySHAP (XGBoost, MSR)" in APPROXIMATORS:
        proxyshap_msr = ProxySHAP(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            adjustment="msr",
        )
        proxyshap_msr.name = "ProxySHAP (XGBoost, MSR)"
        approximator_list.append(proxyshap_msr)
    if "ProxySHAP (XGBoost, MSR, disjoint)" in APPROXIMATORS:
        proxyshap_msr_disjoint = ProxySHAP(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            adjustment="msr",
            disjoint=True,
        )
        proxyshap_msr_disjoint.name = "ProxySHAP (XGBoost, MSR, disjoint)"
        approximator_list.append(proxyshap_msr_disjoint)
    if "ProxySHAP (XGBoost, disjoint)" in APPROXIMATORS:
        proxyshap_disjoint = ProxySHAP(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            adjustment="none",
            disjoint=True,
        )
        proxyshap_disjoint.name = "ProxySHAP (XGBoost, disjoint)"
        approximator_list.append(proxyshap_disjoint)
    SCRATCH_FILE = os.getenv("SCRATCH")
    if SCRATCH_FILE is None:
        SMAC_SAVE_PATH = Path("smac_configurations")
    else:
        SMAC_SAVE_PATH = Path(SCRATCH_FILE) / Path("neurips_tree/hpo")
    if "ProxySHAP* (XGBoost, MSR)" in APPROXIMATORS:
        proxyshap_msr_hpo = ProxySHAPHPO(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            adjustment="msr",
            proxy_model="xgb",
            smac_config_save_path=SMAC_SAVE_PATH,
        )
        proxyshap_msr_hpo.name = "ProxySHAP* (XGBoost, MSR)"
        approximator_list.append(proxyshap_msr_hpo)
    if "ProxySHAP* (XGBoost)" in APPROXIMATORS:
        proxyshap_no_adjust_hpo = ProxySHAPHPO(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            proxy_model="xgb",
            adjustment="none",
            smac_config_save_path=SMAC_SAVE_PATH,
        )
        proxyshap_no_adjust_hpo.name = "ProxySHAP* (XGBoost)"
        approximator_list.append(proxyshap_no_adjust_hpo)
    if "ProxySHAP (Linear)" in APPROXIMATORS:
        proxyshap_linear_no_adjust = ProxySHAP(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            adjustment="none",
            proxy_model=LinearRegression(),
        )
        proxyshap_linear_no_adjust.name = "ProxySHAP (Linear)"
        approximator_list.append(proxyshap_linear_no_adjust)
    if "ProxySHAP (Linear, MSR)" in APPROXIMATORS:
        proxyshap_linear_msr = ProxySHAP(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            adjustment="msr",
            proxy_model=LinearRegression(),
        )
        proxyshap_linear_msr.name = "ProxySHAP (Linear, MSR)"
        approximator_list.append(proxyshap_linear_msr)
    if "ProxySPEX (XGBoost)" in APPROXIMATORS:
        # ProxySpex with XGBoost as value model
        proxy_spex_xgboost = ProxySPEXXGBoost(
            n=NPLAYERS,
            index=INDEX,
            max_order=MAXORDER,
            random_state=RANDOMSTATE,
            top_order=False,
        )
        proxy_spex_xgboost.name = "ProxySPEX (XGBoost)"
        approximator_list.append(proxy_spex_xgboost)
    if "ProxySPEX99 (XGBoost)" in APPROXIMATORS:
        proxy_spex_99_xgboost = ProxySPEXXGBoost(
            n=NPLAYERS,
            index=INDEX,
            max_order=MAXORDER,
            random_state=RANDOMSTATE,
            top_order=False,
            cut_off_quantile=0.99,
        )
        proxy_spex_99_xgboost.name = "ProxySPEX99 (XGBoost)"
        approximator_list.append(proxy_spex_99_xgboost)
    if "ProxySPEX999 (XGBoost)" in APPROXIMATORS:
        proxy_spex_999_xgboost = ProxySPEXXGBoost(
            n=NPLAYERS,
            index=INDEX,
            max_order=MAXORDER,
            random_state=RANDOMSTATE,
            top_order=False,
            cut_off_quantile=0.999,
        )
        proxy_spex_999_xgboost.name = "ProxySPEX999 (XGBoost)"
        approximator_list.append(proxy_spex_999_xgboost)
    if "ProxySPEX9999 (XGBoost)" in APPROXIMATORS:
        proxy_spex_9999_xgboost = ProxySPEXXGBoost(
            n=NPLAYERS,
            index=INDEX,
            max_order=MAXORDER,
            random_state=RANDOMSTATE,
            top_order=False,
            cut_off_quantile=0.9999,
        )
        proxy_spex_9999_xgboost.name = "ProxySPEX9999 (XGBoost)"
        approximator_list.append(proxy_spex_9999_xgboost)
    if "ProxySPEX90 (XGBoost)" in APPROXIMATORS:
        proxy_spex_90_xgboost = ProxySPEXXGBoost(
            n=NPLAYERS,
            index=INDEX,
            max_order=MAXORDER,
            random_state=RANDOMSTATE,
            top_order=False,
            cut_off_quantile=0.90,
        )
        proxy_spex_90_xgboost.name = "ProxySPEX90 (XGBoost)"
        approximator_list.append(proxy_spex_90_xgboost)
    if "ProxySPEX96 (XGBoost)" in APPROXIMATORS:
        proxy_spex_96_xgboost = ProxySPEXXGBoost(
            n=NPLAYERS,
            index=INDEX,
            max_order=MAXORDER,
            random_state=RANDOMSTATE,
            top_order=False,
            cut_off_quantile=0.96,
        )
        proxy_spex_96_xgboost.name = "ProxySPEX96 (XGBoost)"
        approximator_list.append(proxy_spex_96_xgboost)
    if "ProxySPEX97 (XGBoost)" in APPROXIMATORS:
        proxy_spex_97_xgboost = ProxySPEXXGBoost(
            n=NPLAYERS,
            index=INDEX,
            max_order=MAXORDER,
            random_state=RANDOMSTATE,
            top_order=False,
            cut_off_quantile=0.97,
        )
        proxy_spex_97_xgboost.name = "ProxySPEX97 (XGBoost)"
        approximator_list.append(proxy_spex_97_xgboost)
    if "ProxySPEX98 (XGBoost)" in APPROXIMATORS:
        proxy_spex_98_xgboost = ProxySPEXXGBoost(
            n=NPLAYERS,
            index=INDEX,
            max_order=MAXORDER,
            random_state=RANDOMSTATE,
            top_order=False,
            cut_off_quantile=0.98,
        )
        proxy_spex_98_xgboost.name = "ProxySPEX98 (XGBoost)"
        approximator_list.append(proxy_spex_98_xgboost)
    if "ProxySPEX (XGBoost, NoRefinement)" in APPROXIMATORS:
        # ProxySpex with XGBoost as value model and no truncation
        proxy_spex_xgboost_no_truncation = ProxySPEXXGBoostNoRefinement(
            n=NPLAYERS,
            index=INDEX,
            max_order=MAXORDER,
            random_state=RANDOMSTATE,
            top_order=False,
        )
        proxy_spex_xgboost_no_truncation.name = "ProxySPEX (XGBoost, NoRefinement)"
        approximator_list.append(proxy_spex_xgboost_no_truncation)
    if "ProxySPEX (XGBoost, NoTruncation, NoRefinement)" in APPROXIMATORS:
        # ProxySpex with XGBoost as value model, no truncation and no refinement
        proxy_spex_xgboost_no_truncation_no_refinement = ProxySPEXXGBoostNoTruncationNoRefinement(
            n=NPLAYERS,
            index=INDEX,
            max_order=MAXORDER,
            random_state=RANDOMSTATE,
            top_order=False,
        )
        proxy_spex_xgboost_no_truncation_no_refinement.name = (
            "ProxySPEX (XGBoost, NoTruncation, NoRefinement)"
        )
        approximator_list.append(proxy_spex_xgboost_no_truncation_no_refinement)

    # Different Tree Depth
    proxy_model_depth2 = XGBRegressor(max_depth=2, random_state=RANDOMSTATE, tree_method="hist")
    if "ProxySHAP (XGBoost-Depth2)" in APPROXIMATORS:
        proxyshap_depth2 = ProxySHAP(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            adjustment="none",
            proxy_model=proxy_model_depth2,
        )
        proxyshap_depth2.name = "ProxySHAP (XGBoost-Depth2)"
        approximator_list.append(proxyshap_depth2)
    if "ProxySHAP (XGBoost-Depth2, MSR)" in APPROXIMATORS:
        proxyshap_depth2_msr = ProxySHAP(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            adjustment="msr",
            proxy_model=proxy_model_depth2,
        )
        proxyshap_depth2_msr.name = "ProxySHAP (XGBoost-Depth2, MSR)"
        approximator_list.append(proxyshap_depth2_msr)
    if "ProxySPEX (XGBoost-Depth2)" in APPROXIMATORS:
        proxy_spex_depth2 = ProxySPEXXGBoost(
            n=NPLAYERS,
            index=INDEX,
            max_order=MAXORDER,
            random_state=RANDOMSTATE,
            top_order=False,
            proxy_model=proxy_model_depth2,
        )
        proxy_spex_depth2.name = "ProxySPEX (XGBoost-Depth2)"
        approximator_list.append(proxy_spex_depth2)

    proxy_model_depth4 = XGBRegressor(max_depth=4, random_state=RANDOMSTATE, tree_method="hist")
    if "ProxySHAP (XGBoost-Depth4)" in APPROXIMATORS:
        proxyshap_depth4 = ProxySHAP(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            adjustment="none",
            proxy_model=proxy_model_depth4,
        )
        proxyshap_depth4.name = "ProxySHAP (XGBoost-Depth4)"
        approximator_list.append(proxyshap_depth4)
    if "ProxySHAP (XGBoost-Depth4, MSR)" in APPROXIMATORS:
        proxyshap_depth4_msr = ProxySHAP(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            adjustment="msr",
            proxy_model=proxy_model_depth4,
        )
        proxyshap_depth4_msr.name = "ProxySHAP (XGBoost-Depth4, MSR)"
        approximator_list.append(proxyshap_depth4_msr)
    if "ProxySPEX (XGBoost-Depth4)" in APPROXIMATORS:
        proxy_spex_depth4 = ProxySPEXXGBoost(
            n=NPLAYERS,
            index=INDEX,
            max_order=MAXORDER,
            random_state=RANDOMSTATE,
            top_order=False,
            proxy_model=proxy_model_depth4,
        )
        proxy_spex_depth4.name = "ProxySPEX (XGBoost-Depth4)"
        approximator_list.append(proxy_spex_depth4)

    proxy_model_depth6 = XGBRegressor(max_depth=6, random_state=RANDOMSTATE, tree_method="hist")
    if "ProxySHAP (XGBoost-Depth6)" in APPROXIMATORS:
        proxyshap_depth6 = ProxySHAP(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            adjustment="none",
            proxy_model=proxy_model_depth6,
        )
        proxyshap_depth6.name = "ProxySHAP (XGBoost-Depth6)"
        approximator_list.append(proxyshap_depth6)
    if "ProxySHAP (XGBoost-Depth6, MSR)" in APPROXIMATORS:
        proxyshap_depth6_msr = ProxySHAP(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            adjustment="msr",
            proxy_model=proxy_model_depth6,
        )
        proxyshap_depth6_msr.name = "ProxySHAP (XGBoost-Depth6, MSR)"
        approximator_list.append(proxyshap_depth6_msr)
    if "ProxySPEX (XGBoost-Depth6)" in APPROXIMATORS:
        proxy_spex_depth6 = ProxySPEXXGBoost(
            n=NPLAYERS,
            index=INDEX,
            max_order=MAXORDER,
            random_state=RANDOMSTATE,
            top_order=False,
            proxy_model=proxy_model_depth6,
        )
        proxy_spex_depth6.name = "ProxySPEX (XGBoost-Depth6)"
        approximator_list.append(proxy_spex_depth6)

    proxy_model_depth8 = XGBRegressor(max_depth=8, random_state=RANDOMSTATE, tree_method="hist")
    if "ProxySHAP (XGBoost-Depth8)" in APPROXIMATORS:
        proxyshap_depth8 = ProxySHAP(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            adjustment="none",
            proxy_model=proxy_model_depth8,
        )
        proxyshap_depth8.name = "ProxySHAP (XGBoost-Depth8)"
        approximator_list.append(proxyshap_depth8)
    if "ProxySHAP (XGBoost-Depth8, MSR)" in APPROXIMATORS:
        proxyshap_depth8_msr = ProxySHAP(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            adjustment="msr",
            proxy_model=proxy_model_depth8,
        )
        proxyshap_depth8_msr.name = "ProxySHAP (XGBoost-Depth8, MSR)"
        approximator_list.append(proxyshap_depth8_msr)
    if "ProxySPEX (XGBoost-Depth8)" in APPROXIMATORS:
        proxy_spex_depth8 = ProxySPEXXGBoost(
            n=NPLAYERS,
            index=INDEX,
            max_order=MAXORDER,
            random_state=RANDOMSTATE,
            top_order=False,
            proxy_model=proxy_model_depth8,
        )
        proxy_spex_depth8.name = "ProxySPEX (XGBoost-Depth8)"
        approximator_list.append(proxy_spex_depth8)

    if "ProxySPEX (LightGBM)" in APPROXIMATORS:
        # ProxySpex with LightGBM as value model
        proxy_spex_lightgbm = ProxySPEXLightGBM(
            n=NPLAYERS,
            index=INDEX,
            max_order=MAXORDER,
            random_state=RANDOMSTATE,
            top_order=False,
        )
        proxy_spex_lightgbm.name = "ProxySPEX (LightGBM)"
        approximator_list.append(proxy_spex_lightgbm)
    proxy_model_plus = XGBRegressor(
        random_state=RANDOMSTATE,
        n_estimators=2000,
        max_depth=3,
        learning_rate=0.05,
    )
    if "ProxySHAP+ (XGBoost)" in APPROXIMATORS:
        proxyshap_plus_no_adjust = ProxySHAP(
            n=NPLAYERS,
            random_state=RANDOMSTATE,
            pairing_trick=PAIRING,
            index=INDEX,
            max_order=MAXORDER,
            sampling_weights=sampling_weights,
            adjustment="none",
            proxy_model=proxy_model_plus,
        )
        proxyshap_plus_no_adjust.name = "ProxySHAP+ (XGBoost)"
        approximator_list.append(proxyshap_plus_no_adjust)
    if "ProxySPEX+ (XGBoost)" in APPROXIMATORS:
        proxy_spex_plus_xgboost = ProxySPEXXGBoost(
            n=NPLAYERS,
            index=INDEX,
            max_order=MAXORDER,
            random_state=RANDOMSTATE,
            top_order=False,
        )
        proxy_spex_plus_xgboost.name = "ProxySPEX+ (XGBoost)"
        approximator_list.append(proxy_spex_plus_xgboost)

    return approximator_list
