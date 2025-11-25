import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.stats import linregress

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import numpy as np
import math


from scipy.stats import linregress, mannwhitneyu, kruskal


NAME_MAPPING = {
    "vasari asymmetrical ventricles": "Asymmetrical ventricles",
    "vasari enlarged ventricles": "Enlarged ventricles",
    "idh status": "IDH mutation status",
    "vasari f1 tumour location": "Tumor location",
    "vasari f19 ependymal invasion": "Ependymal invasion",
    "vasari f21 deep wm invasion": "Deep WM invasion",
    "vasari f23 cet crosses midline": "CET crosses midline",
    "vasari f4 enhancement quality": "Enhancement quality",
    "vasari f5 proportion enhancing": "Proportion enhancing",
    "vasari f9 multifocal or multicentric": "Multifocal or multicentric",
    "et volume": "ET volume",
    "tumor burden": "Tumor burden",
    "resection status": "Resection status",
    "age": "Age",
    "final pathologic diagnosis (who 2021)": "Pathologic diagnosis (WHO 2021)",
    "mgmt status": "MGMT status",
    "IDH status": "IDH status",
    "max midline shift mm": "Max midline shift (mm)",
    "survival days": "Survival days",
    "vasari f14 proportion of oedema": "Proportion edema",
    "vasari f11 thickness of enhancing margin": "Thickness of enhancing margin",
    "vasari f7 proportion necrosis": "Proportion necrosis",
    "vasari f3 eloquent brain": "Eloquent brain regions invaded",
    "vasari f2 side of tumor epicenter": "Side of tumor epicenter",
    "vasari f24 satellites": "Presence of satellites",
    "vasari f20 cortical involvement": "Cortical involvement",
    "er volume": "Edema volume",
    "nr volume": "Necrotic Core volume",
    "mean distance mm": "Mean midline shift (mm)",
    "p95 distance mm": "95% midline shift (mm)",
    "etiv": "ETIV",
}

# PLOTTING FUNCTION
import matplotlib.pyplot as plt
from scipy.stats import linregress, mannwhitneyu, kruskal
import pandas as pd
import numpy as np
import math

# Muted modern palette
_PALETTE = ["#7B9ACC", "#9ACCA6", "#E4A972", "#D98BA3", "#C2A5CF", "#85C1DC"]


def _p_to_stars(p):
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _add_sig_bar(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#333", linewidth=1.6)
    ax.text((x1 + x2) / 2, y + h * 1.05, text, ha="center", va="bottom", fontsize=10, fontweight="bold")


def plots(df, x_list, y_variable="GBM_Subjects_Spreadsheet__survival_days", only_significant=False, alpha=0.05, save_path=None):

    ncols = 3
    nrows = math.ceil(len(x_list) / ncols)

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})

    # First pass: determine which variables are significant
    significant_vars = []
    y_all = pd.to_numeric(df[y_variable], errors="coerce")

    for x_variable in x_list:
        x_raw = df[x_variable]

        # numeric / boolean detection
        is_bool = pd.api.types.is_bool_dtype(x_raw)
        is_numeric = pd.api.types.is_numeric_dtype(x_raw)

        # try numeric conversion for strings
        if not (is_numeric or is_bool):
            x_num_try = pd.to_numeric(x_raw, errors="coerce")
            is_numeric = x_num_try.notna().mean() > 0.8
        else:
            x_num_try = x_raw

        # numeric branch
        if is_numeric and not is_bool:
            x = pd.to_numeric(x_raw, errors="coerce")
            valid = x.notna() & y_all.notna()

            if valid.sum() > 2:
                slope, intercept, r, p, _ = linregress(x[valid], y_all[valid])
                if p < alpha:
                    significant_vars.append(x_variable)

        # categorical branch
        else:
            tmp = pd.DataFrame({x_variable: x_raw, y_variable: y_all}).dropna()
            if tmp.empty:
                continue

            cats = sorted(tmp[x_variable].unique())
            groups = [tmp.loc[tmp[x_variable] == c, y_variable].values for c in cats]

            if len(groups) == 2:
                g1, g2 = groups
                if len(g1) > 0 and len(g2) > 0:
                    _, p = mannwhitneyu(g1, g2)
                    if p < alpha:
                        significant_vars.append(x_variable)

            elif len(groups) > 2 and all(len(g) > 0 for g in groups):
                _, p = kruskal(*groups)
                if p < alpha:
                    significant_vars.append(x_variable)

    # If filtering is enabled, restrict plots
    if only_significant:
        x_list = [v for v in x_list if v in significant_vars]
        if len(x_list) == 0:
            print("No significant variables found.")
            return

    # recompute layout after filtering
    nrows = math.ceil(len(x_list) / ncols)
    plt.figure(figsize=(8 * ncols, 5 * nrows))

    # ---- second pass: actual plotting ----
    for idx, x_variable in enumerate(x_list, start=1):
        ax = plt.subplot(nrows, ncols, idx)
        x_raw = df[x_variable]

        is_bool = pd.api.types.is_bool_dtype(x_raw)
        is_numeric = pd.api.types.is_numeric_dtype(x_raw)

        if not (is_numeric or is_bool):
            x_num_try = pd.to_numeric(x_raw, errors="coerce")
            is_numeric = x_num_try.notna().mean() > 0.8
        else:
            x_num_try = x_raw

        # numeric predictor
        if is_numeric and not is_bool:
            x = pd.to_numeric(x_raw, errors="coerce")
            y = y_all
            valid = x.notna() & y.notna()
            ax.scatter(x[valid], y[valid], alpha=0.75, s=26, color="#5B8FF9", edgecolor="none")

            if valid.sum() > 2:
                slope, intercept, r, p, _ = linregress(x[valid], y[valid])
                xx = np.linspace(x[valid].min(), x[valid].max(), 100)
                ax.plot(xx, slope * xx + intercept, color="#1F2937", linewidth=2.2)
                stats_text = f"y = {slope:.2f}x + {intercept:.2f}\nR = {r:.2f}\np = {p:.2e}"
                # ax.text(1.02, 0.5, stats_text, transform=ax.transAxes,
                #         fontsize=10, va="center", ha="left",
                #         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#999", alpha=0.95))

                ax.text(
                    0.95,
                    0.95,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=14,  # << bigger font
                    va="top",
                    ha="right",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#555", alpha=0.9),
                )

        # categorical predictor
        else:
            tmp = pd.DataFrame({x_variable: x_raw, y_variable: y_all}).dropna()
            if tmp.empty:
                ax.set_xlabel(x_variable)
                ax.set_ylabel(y_variable)
                continue

            cats = pd.Index(sorted(tmp[x_variable].unique()))
            groups = [tmp.loc[tmp[x_variable] == c, y_variable].values for c in cats]

            bp = ax.boxplot(
                groups,
                tick_labels=[str(c) for c in cats],
                patch_artist=True,
                medianprops=dict(color="#333", linewidth=2),
                boxprops=dict(linewidth=1.2, color="#555"),
                whiskerprops=dict(linewidth=1.2, color="#777"),
                capprops=dict(linewidth=1.2, color="#777"),
                flierprops=dict(marker="o", markersize=3, markerfacecolor="#666", markeredgecolor="none", alpha=0.4),
            )
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(_PALETTE[i % len(_PALETTE)])
                patch.set_alpha(0.6)

            # jittered scatter
            x_positions = np.arange(1, len(groups) + 1)
            for xi, g in zip(x_positions, groups):
                if len(g) == 0:
                    continue
                xj = xi + np.random.uniform(-0.08, 0.08, size=len(g))
                ax.scatter(xj, g, s=10, color="#111", alpha=0.35, linewidths=0)

            ax.tick_params(axis="x", rotation=15)

            # significance bar
            if len(groups) == 2:
                g1, g2 = groups
                if len(g1) > 0 and len(g2) > 0:
                    _, p = mannwhitneyu(g1, g2)
                    stars = _p_to_stars(p)
                    y_max = max(np.max(g1), np.max(g2))
                    y_min = min(np.min(g1), np.min(g2))
                    h = 0.05 * max(1.0, (y_max - y_min))
                    _add_sig_bar(ax, 1, 2, y_max + h, h, stars + f" (p={p:.2e})")

            elif len(groups) > 2 and all(len(g) > 0 for g in groups):
                _, p = kruskal(*groups)
                # optionally annotate

        clean_xvar = " ".join((x_variable.split("__")[-1]).split("_")).lower()
        clean_yvar = " ".join((y_variable.split("__")[-1]).split("_")).lower()

        xv = NAME_MAPPING.get(clean_xvar, clean_xvar.capitalize())
        yv = NAME_MAPPING.get(clean_yvar, clean_yvar.capitalize())

        ax.set_xlabel(xv, fontsize=16)
        ax.set_ylabel(yv, fontsize=16)
        ax.set_title(f"{xv} vs {yv}", fontsize=20)
        # ax.set_title(f"{x_variable.split('brats23_metadata_flattened__global__')[-1]} vs {y_variable}")

        ax.set_facecolor("white")
        ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(h_pad=3.0)
    if save_path is not None:
        fig.savefig(save_path)
    plt.show()


# def plots_survival(
#     df,
#     x_list,
#     duration_col="GBM_Subjects_Spreadsheet__survival_days",
#     event_col="event",
#     only_significant=False,
#     alpha=0.05,
#     save_path=None
# ):
#     """
#     - First pass: test significance using Cox + log-rank
#     - Second pass: draw KM curves + HR boxes
#     """

#     significant_vars = []
#     kmf = KaplanMeierFitter()

#     for x_variable in x_list:
#         tmp = df[[duration_col, event_col, x_variable]].copy().dropna()

#         # skip tiny or no-event groups
#         if tmp.shape[0] < 10 or tmp[event_col].sum() == 0:
#             continue

#         x_raw = tmp[x_variable]

#         # detect type
#         is_bool = pd.api.types.is_bool_dtype(x_raw)
#         is_numeric = pd.api.types.is_numeric_dtype(x_raw)
#         if not (is_numeric or is_bool):
#             maybe = pd.to_numeric(x_raw, errors="coerce")
#             is_numeric = maybe.notna().mean() > 0.8

#         # ---- numeric predictor: Cox on continuous; log-rank on median split ----
#         if is_numeric and not is_bool:
#             tmp[x_variable] = pd.to_numeric(tmp[x_variable], errors="coerce")
#             tmp = tmp.dropna()

#             try:
#                 # Cox
#                 cph = CoxPHFitter()
#                 cph.fit(tmp[[duration_col, event_col, x_variable]],
#                         duration_col=duration_col,
#                         event_col=event_col)
#                 p_cox = float(cph.summary.loc[x_variable, "p"])
#             except:
#                 continue

#             # log-rank (Low vs High)
#             median_val = tmp[x_variable].median()
#             grp = np.where(tmp[x_variable] <= median_val, 0, 1)
#             try:
#                 lr = logrank_test(
#                     tmp.loc[grp==0, duration_col],
#                     tmp.loc[grp==1, duration_col],
#                     tmp.loc[grp==0, event_col],
#                     tmp.loc[grp==1, event_col]
#                 )
#                 p_lr = lr.p_value
#             except:
#                 p_lr = 1.0

#             if min(p_cox, p_lr) < alpha:
#                 significant_vars.append(x_variable)

#         # ---- categorical predictor ----
#         else:
#             cats = tmp[x_variable].value_counts()
#             if len(cats) < 2:
#                 continue

#             # log-rank across groups
#             try:
#                 lr_multi = multivariate_logrank_test(
#                     tmp[duration_col], tmp[x_variable], tmp[event_col]
#                 )
#                 p_lr = lr_multi.p_value
#             except:
#                 p_lr = 1.0

#             if p_lr < alpha:
#                 significant_vars.append(x_variable)

#     # filter if needed
#     if only_significant:
#         x_list = [v for v in x_list if v in significant_vars]
#         if not x_list:
#             print("No significant survival associations found.")
#             return

#     ncols = 3
#     nrows = math.ceil(len(x_list) / ncols)
#     fig = plt.figure(figsize=(8 * ncols, 5 * nrows))

#     for idx, x_variable in enumerate(x_list, start=1):
#         ax = plt.subplot(nrows, ncols, idx)
#         tmp = df[[duration_col, event_col, x_variable]].copy().dropna()

#         if tmp.shape[0] < 10 or tmp[event_col].sum() == 0:
#             ax.set_visible(False)
#             continue

#         x_raw = tmp[x_variable]
#         is_bool = pd.api.types.is_bool_dtype(x_raw)
#         is_numeric = pd.api.types.is_numeric_dtype(x_raw)
#         if not (is_numeric or is_bool):
#             maybe = pd.to_numeric(x_raw, errors="coerce")
#             is_numeric = maybe.notna().mean() > 0.8

#         stats_text = ""

#         #  numeric predictor
#         if is_numeric and not is_bool:
#             tmp[x_variable] = pd.to_numeric(tmp[x_variable], errors="coerce")
#             tmp = tmp.dropna()

#             median_val = tmp[x_variable].median()
#             tmp["group"] = np.where(tmp[x_variable] <= median_val, "Low", "High")

#             colors = _PALETTE
#             for i, grp in enumerate(["Low", "High"]):
#                 mask = tmp["group"] == grp
#                 if mask.sum() < 5:
#                     continue
#                 kmf.fit(tmp.loc[mask, duration_col],
#                         event_observed=tmp.loc[mask, event_col],
#                         label=f"{grp} (n={mask.sum()})")
#                 kmf.plot(ax=ax, ci_show=False, linewidth=2.0,
#                          color=colors[i % len(colors)])

#             # log-rank
#             g_low = tmp["group"] == "Low"
#             g_high = tmp["group"] == "High"
#             lr = logrank_test(
#                 tmp.loc[g_low, duration_col],
#                 tmp.loc[g_high, duration_col],
#                 tmp.loc[g_low, event_col],
#                 tmp.loc[g_high, event_col]
#             )
#             p_lr = lr.p_value

#             # Cox
#             try:
#                 cph = CoxPHFitter()
#                 cph.fit(tmp[[duration_col, event_col, x_variable]],
#                         duration_col=duration_col,
#                         event_col=event_col)
#                 row = cph.summary.loc[x_variable]
#                 stats_text = (
#                     f"HR={row['exp(coef)']:.2f} "
#                     f"[{row['exp(coef) lower 95%']:.2f}, {row['exp(coef) upper 95%']:.2f}]\n"
#                     f"p(Cox)={row['p']:.2e}\n"
#                     f"p(log-rank)={p_lr:.2e}"
#                 )
#             except:
#                 stats_text = f"p(log-rank)={p_lr:.2e}"

#         #  categorical predictor
#         else:
#             cats = sorted(tmp[x_variable].unique(), key=lambda x: str(x))
#             colors = _PALETTE
#             for i, cat in enumerate(cats):
#                 mask = tmp[x_variable] == cat
#                 if mask.sum() < 5:
#                     continue
#                 kmf.fit(tmp.loc[mask, duration_col],
#                         event_observed=tmp.loc[mask, event_col],
#                         label=f"{cat} (n={mask.sum()})")
#                 kmf.plot(ax=ax, ci_show=False, linewidth=2.0,
#                          color=colors[i % len(colors)])

#             # log-rank across groups
#             try:
#                 lr_multi = multivariate_logrank_test(
#                     tmp[duration_col], tmp[x_variable], tmp[event_col]
#                 )
#                 p_lr = lr_multi.p_value
#             except:
#                 p_lr = 1.0

#             stats_text = f"p(log-rank)={p_lr:.2e}"


#         #  aesthetics
#         ax.set_xlabel("Time (days)", fontsize=20)
#         ax.set_ylabel("Survival probability", fontsize=20)


#         clean_xvar=' '.join((x_variable.split('__')[-1]).split('_')).lower()
#         xv = NAME_MAPPING.get(clean_xvar, clean_xvar.capitalize())

#         ax.set_title(xv, fontsize=20)
#         ax.tick_params(axis="both", labelsize=18)
#         ax.grid(True, axis="both", color="#E6E6E6", linewidth=0.6)
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         ax.legend(fontsize=14, loc="center right")

#         # ax.text(1.02, 0.5, stats_text,
#         #         transform=ax.transAxes, fontsize=9, va="center", ha="left",
#         #         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.95))

#         ax.text(
#         0.95, 0.95, stats_text,
#         transform=ax.transAxes,
#         fontsize=14,             # << bigger font
#         va="top", ha="right",
#         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#555", alpha=0.9)
#         )

#     plt.tight_layout(h_pad=3.0)
#     if save_path is not None:
#         fig.savefig(save_path)
#     plt.show()


def plots_survival(
    df,
    x_list,
    duration_col="GBM_Subjects_Spreadsheet__survival_days",
    event_col="event",
    only_significant=False,
    alpha=0.05,
    sort=False,
    save_path=None,
):
    """
    Sort plots by p(log-rank), but still compute CoxPH + HR for numeric predictors.
    """

    kmf = KaplanMeierFitter()

    # store log-rank p-values for sorting
    plogrank = {}

    # --------------------------- FIRST PASS ----------------------------------
    for x_variable in x_list:
        tmp = df[[duration_col, event_col, x_variable]].copy().dropna()

        # skip tiny or no-event groups
        if tmp.shape[0] < 10 or tmp[event_col].sum() == 0:
            plogrank[x_variable] = 1.0
            continue

        x_raw = tmp[x_variable]

        # detect type
        is_bool = pd.api.types.is_bool_dtype(x_raw)
        is_numeric = pd.api.types.is_numeric_dtype(x_raw)
        if not (is_numeric or is_bool):
            maybe = pd.to_numeric(x_raw, errors="coerce")
            is_numeric = maybe.notna().mean() > 0.8

        # ---- NUMERIC predictor: median split log-rank ----
        if is_numeric and not is_bool:
            tmp[x_variable] = pd.to_numeric(tmp[x_variable], errors="coerce")
            tmp = tmp.dropna()

            median_val = tmp[x_variable].median()
            grp = np.where(tmp[x_variable] <= median_val, 0, 1)

            try:
                lr = logrank_test(
                    tmp.loc[grp == 0, duration_col], tmp.loc[grp == 1, duration_col], tmp.loc[grp == 0, event_col], tmp.loc[grp == 1, event_col]
                )
                p_lr = float(lr.p_value)
            except:
                p_lr = 1.0

            # store log-rank p only
            plogrank[x_variable] = p_lr

        # ---- CATEGORICAL predictor ----
        else:
            try:
                lr_multi = multivariate_logrank_test(tmp[duration_col], tmp[x_variable], tmp[event_col])
                p_lr = float(lr_multi.p_value)
            except:
                p_lr = 1.0

            plogrank[x_variable] = p_lr

    # --------------------------- FILTER / SORT ----------------------------------

    if only_significant:
        x_list = [v for v in x_list if plogrank.get(v, 1.0) < alpha]
        if not x_list:
            print("No significant survival associations found.")
            return

    # SORT BY log-rank p-value ASCENDING (most significant first)
    if sort:
        x_list = sorted(x_list, key=lambda v: plogrank.get(v, 1.0))

    # --------------------------- PLOTTING PASS ----------------------------------

    ncols = 3
    nrows = math.ceil(len(x_list) / ncols)
    fig = plt.figure(figsize=(8 * ncols, 5 * nrows))

    for idx, x_variable in enumerate(x_list, start=1):
        ax = plt.subplot(nrows, ncols, idx)
        tmp = df[[duration_col, event_col, x_variable]].copy().dropna()

        if tmp.shape[0] < 10 or tmp[event_col].sum() == 0:
            ax.set_visible(False)
            continue

        x_raw = tmp[x_variable]
        is_bool = pd.api.types.is_bool_dtype(x_raw)
        is_numeric = pd.api.types.is_numeric_dtype(x_raw)

        if not (is_numeric or is_bool):
            maybe = pd.to_numeric(x_raw, errors="coerce")
            is_numeric = maybe.notna().mean() > 0.8

        stats_text = ""

        # --------------------- NUMERIC PREDICTOR ---------------------
        if is_numeric and not is_bool:
            tmp[x_variable] = pd.to_numeric(tmp[x_variable], errors="coerce")
            tmp = tmp.dropna()

            median_val = tmp[x_variable].median()
            tmp["group"] = np.where(tmp[x_variable] <= median_val, "Low", "High")

            colors = _PALETTE

            # KM curves
            for i, grp in enumerate(["Low", "High"]):
                mask = tmp["group"] == grp
                if mask.sum() < 5:
                    continue
                kmf.fit(tmp.loc[mask, duration_col], event_observed=tmp.loc[mask, event_col], label=f"{grp} (n={mask.sum()})")
                kmf.plot(ax=ax, ci_show=False, linewidth=2.0, color=colors[i % len(colors)])

            # log-rank
            g_low = tmp["group"] == "Low"
            g_high = tmp["group"] == "High"
            lr = logrank_test(tmp.loc[g_low, duration_col], tmp.loc[g_high, duration_col], tmp.loc[g_low, event_col], tmp.loc[g_high, event_col])
            p_lr = float(lr.p_value)

            # CoxPH (still computed, not used for sorting)
            try:
                cph = CoxPHFitter()
                cph.fit(tmp[[duration_col, event_col, x_variable]], duration_col=duration_col, event_col=event_col)
                row = cph.summary.loc[x_variable]
                stats_text = (
                    f"HR={row['exp(coef)']:.2f} "
                    f"[{row['exp(coef) lower 95%']:.2f}, {row['exp(coef) upper 95%']:.2f}]\n"
                    f"p(Cox)={row['p']:.2e}\n"
                    f"p(log-rank)={p_lr:.2e}"
                )
            except:
                stats_text = f"p(log-rank)={p_lr:.2e}"

        # --------------------- CATEGORICAL PREDICTOR ---------------------
        else:
            cats = sorted(tmp[x_variable].unique(), key=lambda x: str(x))
            colors = _PALETTE

            for i, cat in enumerate(cats):
                mask = tmp[x_variable] == cat
                if mask.sum() < 5:
                    continue
                kmf.fit(tmp.loc[mask, duration_col], event_observed=tmp.loc[mask, event_col], label=f"{cat} (n={mask.sum()})")
                kmf.plot(ax=ax, ci_show=False, linewidth=2.0, color=colors[i % len(colors)])

            try:
                lr_multi = multivariate_logrank_test(tmp[duration_col], tmp[x_variable], tmp[event_col])
                p_lr = float(lr_multi.p_value)
            except:
                p_lr = 1.0

            stats_text = f"p(log-rank)={p_lr:.2e}"

        # ------------------ Aesthetics -----------------------
        ax.set_xlabel("Time (days)", fontsize=20)
        ax.set_ylabel("Survival probability", fontsize=20)

        clean_xvar = " ".join((x_variable.split("__")[-1]).split("_")).lower()
        xv = NAME_MAPPING.get(clean_xvar, clean_xvar.capitalize())
        ax.set_title(xv, fontsize=20)

        ax.tick_params(axis="both", labelsize=18)
        ax.grid(True, axis="both", color="#E6E6E6", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=14, loc="center right")

        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=14,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#555", alpha=0.9),
        )

    plt.tight_layout(h_pad=3.0)
    if save_path is not None:
        fig.savefig(save_path)
    plt.show()
