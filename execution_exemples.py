import csv
import json
import math
from pathlib import Path
from branch_and_bound import *
from cuts import *
from presolve import *
from primal_heuristics import *
from branching import *
from node_selection import *

PROBLEMS_DIR = Path("problems_nl")
JSON_DIR     = Path("results_json")
TABLES_DIR   = Path("results_tables")
LATEX_DIR    = Path("results_latex")

# =============================================================================
# CONFIGURAÇÕES DE B&B (0–9)
# =============================================================================

def get_all_configs() -> dict:
    """
    Retorna um dicionário {nome_config: função build_solver(model_path)}.
    Cada função retorna um BranchAndBound configurado.
    """

    def cfg0(model_path):
        """Baseline cru"""
        return BranchAndBound(
            model_path=model_path,
            strategy=BreadthFirst(),
            direction=BranchDirection.CEIL_FIRST,
            branch_rule=FirstFractionalRule(),
            verbose=False,
        )

    def cfg1(model_path):
        """Presolve"""
        return BranchAndBound(
            model_path=model_path,
            strategy=BreadthFirst(),
            direction=BranchDirection.CEIL_FIRST,
            branch_rule=FirstFractionalRule(),
            presolver=Presolver(
                rules=[
                    BoundTightening(),
                    RowFeasibilityAndRedundancy(),
                    SubstituteFixedVariables(),
                    ImplicitFreeVariableDetection(),
                    SingletonRowReduction(),
                    EuclideanReduction(),
                    CoefficientStrengthening(),
                    ChvatalGomoryStrengthening(),
                    ParallelRows(),
                ],
                verbose=False,
            ),
            verbose=False,
        )

    def cfg2(model_path):
        """Depth-First"""
        return BranchAndBound(
            model_path=model_path,
            strategy=DepthFirst(),
            direction=BranchDirection.CEIL_FIRST,
            branch_rule=FirstFractionalRule(),
            verbose=False,
        )

    def cfg3(model_path):
        """Best-First"""
        return BranchAndBound(
            model_path=model_path,
            strategy=BestFirst(),
            direction=BranchDirection.CEIL_FIRST,
            branch_rule=FirstFractionalRule(),
            verbose=False,
        )

    def cfg4(model_path):
        """StrongBranchingRule"""
        return BranchAndBound(
            model_path=model_path,
            strategy=BreadthFirst(),
            direction=BranchDirection.CEIL_FIRST,
            branch_rule=StrongBranchingRule(max_candidates=8, score_mode="product"),
            verbose=False,
        )

    def cfg5(model_path):
        """PseudoCostRule"""
        return BranchAndBound(
            model_path=model_path,
            strategy=BreadthFirst(),
            direction=BranchDirection.CEIL_FIRST,
            branch_rule=PseudoCostRule(),
            verbose=False,
        )

    def cfg6(model_path):
        """Cutting Planes"""
        return BranchAndBound(
            model_path=model_path,
            strategy=BreadthFirst(),
            direction=BranchDirection.CEIL_FIRST,
            branch_rule=FirstFractionalRule(),
            cutting_planes=[
                MixedIntegerGomoryCut(only_root=True, max_cuts=20),
                SimpleMIR(only_root=True, max_cuts=20),
                LiftingExtending(only_root=True, max_cuts=20),
                CoverCut(only_root=True, max_cuts=10),
            ],
            verbose=False,
        )

    def cfg7(model_path):
        """Heurísticas completas"""
        return BranchAndBound(
            model_path=model_path,
            strategy=BreadthFirst(),
            direction=BranchDirection.CEIL_FIRST,
            branch_rule=FirstFractionalRule(),
            heuristics=[
                RoundingHeuristic(only_root=False, probability=1.0),
                FixAndPropagate(only_root=False, probability=1.0),
                RENS(only_root=True, probability=1.0),
                RINS(only_root=False, probability=0.3),
                LocalSearch(only_root=False, probability=0.3),
                LocalBranching(only_root=False, probability=0.3),
                LNS(FractionalGuidedDestroy(), probability=0.3),
                LNS(RandomDestroy(), probability=0.2),
                LNS(ConstraintBasedDestroy(), probability=0.2),
                LNS(ObjectiveBasedDestroy(), probability=0.2),
                ALNS(probability=0.3),
                ALNS(acceptance=ALNSAcceptance.SIMULATED_ANNEAL, probability=0.2),
                FractionalDiving(only_root=False, probability=0.3),
                GuidedDiving(only_root=False, probability=0.3),
                LockDiving(only_root=False, probability=0.3),
                FeasibilityPump(only_root=False, probability=0.3),
                FeasibilityJump(only_root=False, probability=0.3),
            ],
            cutting_planes=[
                MixedIntegerGomoryCut(only_root=True, max_cuts=20),
                SimpleMIR(only_root=True, max_cuts=20),
                LiftingExtending(only_root=True, max_cuts=20),
                CoverCut(only_root=True, max_cuts=10),
            ],
            verbose=False,
        )

    def cfg8(model_path):
        """Completo (produção)"""
        return BranchAndBound(
            model_path=model_path,
            strategy=DepthFirst(),
            direction=BranchDirection.CEIL_FIRST,
            branch_rule=PseudoCostRule(),
            presolver=Presolver(
                rules=[
                    BoundTightening(),
                    RowFeasibilityAndRedundancy(),
                    SubstituteFixedVariables(),
                    ImplicitFreeVariableDetection(),
                    SingletonRowReduction(),
                    EuclideanReduction(),
                    CoefficientStrengthening(),
                    ChvatalGomoryStrengthening(),
                    ParallelRows(),
                ],
                verbose=False,
            ),
            heuristics=[
                RoundingHeuristic(only_root=False, probability=1.0),
                FixAndPropagate(only_root=False, probability=0.3),
                RINS(only_root=False, probability=0.3),
                FractionalDiving(only_root=False, probability=0.3),
                FeasibilityPump(only_root=False, probability=0.3),
            ],
            cutting_planes=[
                MixedIntegerGomoryCut(only_root=True, max_cuts=20),
                SimpleMIR(only_root=True, max_cuts=20),
                LiftingExtending(only_root=True, max_cuts=20),
                CoverCut(only_root=True, max_cuts=10),
            ],
            verbose=False,
        )

    return {
        "cfg0_baseline":      cfg0,
        "cfg1_presolve":      cfg1,
        "cfg2_depth_first":   cfg2,
        "cfg3_best_first":    cfg3,
        "cfg4_strong_branch": cfg4,
        "cfg5_pseudocost":    cfg5,
        "cfg6_cuts":          cfg6,
        "cfg7_heuristics":    cfg7,
        "cfg8_complete":      cfg8,
    }

# =============================================================================
# UTILITÁRIOS (inalterados)
# =============================================================================

def safe_get(dct, *keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def is_bad_number(x):
    return isinstance(x, float) and (math.isinf(x) or math.isnan(x))

def round_or_none(x, ndigits=6):
    if x is None:
        return None
    if is_bad_number(x):
        return None
    if isinstance(x, float):
        return round(x, ndigits)
    return x

def fmt_csv_value(x):
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isinf(x) or math.isnan(x):

            return ""
    return x

def fmt_latex_value(x, ndigits=4):
    if x is None:
        return "--"
    if isinstance(x, float):
        if math.isinf(x) or math.isnan(x):
            return "--"
        return f"{x:.{ndigits}f}"
    return str(x)

def latex_escape(text):
    if text is None:
        return "--"
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&":  r"\&",
        "%":  r"\%",
        "$":  r"\$",
        "#":  r"\#",
        "_":  r"\_",
        "{":  r"\{",
        "}":  r"\}",
        "~":  r"\textasciitilde{}",
        "^":  r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s

def compute_root_gap(obj_value, lp_value):
    if obj_value is None or lp_value is None:
        return None
    if is_bad_number(obj_value) or is_bad_number(lp_value):
        return None
    if abs(obj_value) < 1e-12:
        return None
    return max(0.0, (obj_value - lp_value) / abs(obj_value))

def sum_heuristic_metric(stats, metric_name):
    heur = stats.get("heuristics", {})
    total = 0
    for _, hdata in heur.items():
        total += hdata.get(metric_name, 0)
    return total

def sum_cut_metric(stats, metric_name):
    cuts = stats.get("cuts", {})
    total = 0
    for _, cdata in cuts.items():
        total += cdata.get(metric_name, 0)
    return total

def presolve_rule_is_active(rule_data):
    return any([
        rule_data.get("bounds_tightened", 0) > 0,
        rule_data.get("rows_removed", 0) > 0,
        rule_data.get("vars_fixed", 0) > 0,
        rule_data.get("coefficients_changed", 0) > 0,
    ])

# =============================================================================
# RODAR INSTÂNCIAS — agora por configuração
# =============================================================================

def run_all_instances_for_config(
    config_name: str,
    build_fn,
    problems_dir: Path,
    json_base_dir: Path,
):
    """
    Roda todas as instâncias .mps para uma configuração específica.
    JSONs salvos em json_base_dir / config_name / <instancia>.json
    """
    json_dir = json_base_dir / config_name
    json_dir.mkdir(parents=True, exist_ok=True)

    mps_files = sorted(problems_dir.glob("*.mps"))
    print(f"\n  [{config_name}] {len(mps_files)} instâncias encontradas.")

    for mps_file in mps_files:
        out_json = json_dir / f"{mps_file.stem}.json"

        # Pula se já foi rodado (útil para reruns parciais)
        if out_json.exists():
            print(f"    SKIP (já existe): {mps_file.name}")
            continue

        print(f"    Rodando: {mps_file.name} ...", end=" ", flush=True)
        try:
            bb = build_fn(str(mps_file))
            bb.solve()
            bb.export_stats(str(out_json))
            print("OK")
        except Exception as e:
            print(f"ERRO: {e}")

def run_all_configs(
    problems_dir="problems_nl",
    json_base_dir="results_json",
):
    configs = get_all_configs()
    problems_path = Path(problems_dir)
    json_base_path = Path(json_base_dir)

    print("=" * 70)
    print("ETAPA 1: Rodando todas as configurações")
    print("=" * 70)

    for config_name, build_fn in configs.items():
        print(f"\n>>> Configuração: {config_name}")
        run_all_instances_for_config(
            config_name=config_name,
            build_fn=build_fn,
            problems_dir=problems_path,
            json_base_dir=json_base_path,
        )

# =============================================================================
# EXTRAÇÃO DE MÉTRICAS (inalterada)
# =============================================================================

def extract_metrics_from_json(json_file: Path) -> dict:
    with open(json_file, "r", encoding="utf-8") as f:
        stats = json.load(f)

    instance_name = json_file.stem
    n_vars        = safe_get(stats, "instance", "n_vars")
    n_constraints = safe_get(stats, "instance", "n_constraints")
    status        = safe_get(stats, "final", "status")
    obj_value     = safe_get(stats, "final", "obj_value")
    root_lp       = safe_get(stats, "root_lp", "lp_value")
    root_lp_after_cuts = safe_get(stats, "root_lp", "lp_value_after_cuts")

    gap_root = gap_root_after_cuts = None

    if status == "optimal" and obj_value is not None:
        gap_root            = compute_root_gap(obj_value, root_lp)
        gap_root_after_cuts = compute_root_gap(obj_value, root_lp_after_cuts)

    runtime_total      = safe_get(stats, "final", "runtime_total")
    runtime_lp         = safe_get(stats, "final", "runtime_lp")
    runtime_heuristics = safe_get(stats, "final", "runtime_heuristics")
    runtime_cuts       = safe_get(stats, "final", "runtime_cuts")

    nodes_explored        = safe_get(stats, "tree", "nodes_explored")
    nodes_pruned          = safe_get(stats, "tree", "nodes_pruned")
    pruned_by_infeasibility = safe_get(stats, "tree", "pruned_by_infeasibility")
    pruned_by_bound       = safe_get(stats, "tree", "pruned_by_bound")
    max_depth             = safe_get(stats, "tree", "max_depth")
    max_open_nodes        = safe_get(stats, "tree", "max_open_nodes")

    n_incumbents          = len(stats.get("incumbents", []))
    total_heur_calls      = sum_heuristic_metric(stats, "calls")
    total_heur_improvements = sum_heuristic_metric(stats, "improvements")
    total_heur_solutions  = sum_heuristic_metric(stats, "solutions_found")
    total_cuts_added      = sum_cut_metric(stats, "cuts_added")

    presolve_summary   = stats.get("presolve", {}).get("summary", {})
    presolve_runtime   = presolve_summary.get("total_runtime")
    presolve_bounds    = presolve_summary.get("total_bounds_tightened")
    presolve_rows_removed = presolve_summary.get("total_rows_removed")
    presolve_vars_fixed   = presolve_summary.get("total_vars_fixed")
    presolve_coeff_changed = presolve_summary.get("total_coefficients_changed")
    presolve_n_rules   = presolve_summary.get("n_rules")

    pct_rows_removed = pct_vars_fixed = pct_presolve_time = None

    if n_constraints not in (None, 0) and presolve_rows_removed is not None:
        pct_rows_removed = 100.0 * presolve_rows_removed / n_constraints
    if n_vars not in (None, 0) and presolve_vars_fixed is not None:
        pct_vars_fixed = 100.0 * presolve_vars_fixed / n_vars
    if runtime_total not in (None, 0) and presolve_runtime is not None:
        pct_presolve_time = 100.0 * presolve_runtime / runtime_total

    return {
        "instance":                   instance_name,
        "status":                     status,
        "obj_value":                  round_or_none(obj_value),
        "root_lp":                    round_or_none(root_lp),
        "root_lp_after_cuts":         round_or_none(root_lp_after_cuts),
        "gap_root":                   round_or_none(gap_root),
        "gap_root_after_cuts":        round_or_none(gap_root_after_cuts),
        "runtime_total":              round_or_none(runtime_total),
        "runtime_lp":                 round_or_none(runtime_lp),
        "runtime_heuristics":         round_or_none(runtime_heuristics),
        "runtime_cuts":               round_or_none(runtime_cuts),
        "nodes_explored":             nodes_explored,
        "nodes_pruned":               nodes_pruned,
        "pruned_by_infeasibility":    pruned_by_infeasibility,
        "pruned_by_bound":            pruned_by_bound,
        "max_depth":                  max_depth,
        "max_open_nodes":             max_open_nodes,
        "n_incumbents":               n_incumbents,
        "total_heur_calls":           total_heur_calls,
        "total_heur_improvements":    total_heur_improvements,
        "total_heur_solutions":       total_heur_solutions,
        "total_cuts_added":           total_cuts_added,
        "presolve_n_rules":           presolve_n_rules,
        "presolve_bounds_tightened":  presolve_bounds,
        "presolve_rows_removed":      presolve_rows_removed,
        "presolve_vars_fixed":        presolve_vars_fixed,
        "presolve_coefficients_changed": presolve_coeff_changed,
        "presolve_runtime":           round_or_none(presolve_runtime),
        "pct_rows_removed":           round_or_none(pct_rows_removed),
        "pct_vars_fixed":             round_or_none(pct_vars_fixed),
        "pct_presolve_time":          round_or_none(pct_presolve_time),
    }

def collect_all_metrics(json_dir: Path) -> list:
    rows = []
    for jf in sorted(json_dir.glob("*.json")):
        try:
            rows.append(extract_metrics_from_json(jf))
        except Exception as e:
            print(f"  Erro lendo {jf.name}: {e}")
    return rows

def collect_presolve_rule_rows(json_dir: Path) -> list:
    rows = []
    for jf in sorted(json_dir.glob("*.json")):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                stats = json.load(f)
            instance_name = jf.stem
            rules = safe_get(stats, "presolve", "rules", default=[])
            for rule_data in rules:
                rows.append({
                    "instance":              instance_name,
                    "rule":                  rule_data.get("rule"),
                    "bounds_tightened":      rule_data.get("bounds_tightened"),
                    "rows_removed":          rule_data.get("rows_removed"),
                    "vars_fixed":            rule_data.get("vars_fixed"),
                    "coefficients_changed":  rule_data.get("coefficients_changed"),
                    "runtime":               round_or_none(rule_data.get("runtime")),
                    "active":    "yes" if presolve_rule_is_active(rule_data) else "no",
                    "infeasible": "yes" if rule_data.get("infeasible", False) else "no",
                })
        except Exception as e:
            print(f"  Erro lendo presolve de {jf.name}: {e}")
    return rows

# =============================================================================
# ESCRITA DE TABELAS (inalterada)
# =============================================================================

def write_csv(filepath: Path, fieldnames, rows):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: fmt_csv_value(row.get(k)) for k in fieldnames}
            writer.writerow(out)

def write_latex_table(filepath: Path, caption: str, label: str, fieldnames, rows):
    colspec = "l" + "r" * (len(fieldnames) - 1)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{latex_escape(caption)}}}\n")
        f.write(f"\\label{{{latex_escape(label)}}}\n")
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n")
        f.write("\\hline\n")
        header = " & ".join(latex_escape(h) for h in fieldnames) + " \\\\\n"
        f.write(header)
        f.write("\\hline\n")
        for row in rows:
            vals = []
            for k in fieldnames:
                v = row.get(k)
                if k in ("instance", "status", "rule", "active", "infeasible", "config"):
                    vals.append(latex_escape(v))
                else:
                    vals.append(latex_escape(fmt_latex_value(v)))
            f.write(" & ".join(vals) + " \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

# =============================================================================
# GERAR TABELAS POR CONFIGURAÇÃO + TABELA COMPARATIVA
# =============================================================================

def generate_tables_for_config(
    config_name: str,
    rows: list,
    presolve_rule_rows: list,
    tables_base: Path,
    latex_base: Path,
):
    """Gera as tabelas padrão (1–4 + presolve) para uma configuração."""
    tables_path = tables_base / config_name
    latex_path  = latex_base  / config_name
    tables_path.mkdir(parents=True, exist_ok=True)
    latex_path.mkdir(parents=True, exist_ok=True)

    # Tabela 1 — Relaxação
    t1 = ["instance","status","obj_value","root_lp","root_lp_after_cuts","gap_root","gap_root_after_cuts"]
    write_csv(tables_path / "table_1_resultado_relaxacao.csv", t1,
              [{k: r.get(k) for k in t1} for r in rows])
    write_latex_table(latex_path / "table_1_resultado_relaxacao.tex",
                      f"[{config_name}] Resultado final e qualidade da relaxação.",
                      f"tab:{config_name}_relaxacao", t1,
                      [{k: r.get(k) for k in t1} for r in rows])

    # Tabela 2 — Árvore
    t2 = ["instance","nodes_explored","nodes_pruned","pruned_by_infeasibility",
          "pruned_by_bound","max_depth","max_open_nodes"]
    write_csv(tables_path / "table_2_arvore.csv", t2,
              [{k: r.get(k) for k in t2} for r in rows])
    write_latex_table(latex_path / "table_2_arvore.tex",
                      f"[{config_name}] Tamanho e esforço da árvore B&B.",
                      f"tab:{config_name}_arvore", t2,
                      [{k: r.get(k) for k in t2} for r in rows])

    # Tabela 3 — Tempos
    t3 = ["instance","runtime_total","runtime_lp","runtime_heuristics","runtime_cuts"]
    write_csv(tables_path / "table_3_tempos.csv", t3,
              [{k: r.get(k) for k in t3} for r in rows])
    write_latex_table(latex_path / "table_3_tempos.tex",
                      f"[{config_name}] Tempos de execução por componente.",
                      f"tab:{config_name}_tempos", t3,
                      [{k: r.get(k) for k in t3} for r in rows])

    # Tabela 4 — Heurísticas e cortes
    t4 = ["instance","n_incumbents","total_heur_calls","total_heur_improvements",
          "total_heur_solutions","total_cuts_added"]
    write_csv(tables_path / "table_4_heuristicas_cortes.csv", t4,
              [{k: r.get(k) for k in t4} for r in rows])
    write_latex_table(latex_path / "table_4_heuristicas_cortes.tex",
                      f"[{config_name}] Resumo de heurísticas e cortes.",
                      f"tab:{config_name}_heuristicas", t4,
                      [{k: r.get(k) for k in t4} for r in rows])

    # Tabela P1 — Presolve summary
    tp1 = ["instance","presolve_n_rules","presolve_bounds_tightened","presolve_rows_removed",
           "presolve_vars_fixed","presolve_coefficients_changed","presolve_runtime",
           "pct_rows_removed","pct_vars_fixed","pct_presolve_time"]
    write_csv(tables_path / "table_p1_presolve_summary.csv", tp1,
              [{k: r.get(k) for k in tp1} for r in rows])
    write_latex_table(latex_path / "table_p1_presolve_summary.tex",
                      f"[{config_name}] Resumo do presolve por instância.",
                      f"tab:{config_name}_presolve_summary", tp1,
                      [{k: r.get(k) for k in tp1} for r in rows])

    # Tabela P2 — Presolve rules
    tp2 = ["instance","rule","bounds_tightened","rows_removed","vars_fixed",
           "coefficients_changed","runtime","active","infeasible"]
    write_csv(tables_path / "table_p2_presolve_rules.csv", tp2,
              [{k: r.get(k) for k in tp2} for r in presolve_rule_rows])
    write_latex_table(latex_path / "table_p2_presolve_rules.tex",
                      f"[{config_name}] Desempenho das regras de presolve.",
                      f"tab:{config_name}_presolve_rules", tp2,
                      [{k: r.get(k) for k in tp2} for r in presolve_rule_rows])

def generate_comparison_table(
    all_rows_by_config: dict,
    tables_base: Path,
    latex_base: Path,
):
    """
    Gera uma tabela comparativa com uma linha por (config, instância),
    mostrando as métricas-chave lado a lado.
    """
    tables_base.mkdir(parents=True, exist_ok=True)
    latex_base.mkdir(parents=True, exist_ok=True)

    fields = ["config", "instance", "status", "obj_value",
              "runtime_total", "nodes_explored", "total_cuts_added",
              "total_heur_solutions", "gap_root"]

    flat_rows = []
    for config_name, rows in all_rows_by_config.items():
        for row in rows:
            flat_row = {k: row.get(k) for k in fields}
            flat_row["config"] = config_name
            flat_rows.append(flat_row)

    write_csv(tables_base / "comparison_all_configs.csv", fields, flat_rows)
    write_latex_table(
        latex_base / "comparison_all_configs.tex",
        "Comparação entre configurações de B\\&B.",
        "tab:comparison_all_configs",
        fields,
        flat_rows,
    )
    print(f"  Tabela comparativa salva em: {tables_base / 'comparison_all_configs.csv'}")

# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_experiments_and_generate_tables(
    problems_dir="problems_nl",
    json_base_dir="results_json",
    tables_base_dir="results_tables",
    latex_base_dir="results_latex",
):
    configs         = get_all_configs()
    problems_path   = Path(problems_dir)
    json_base_path  = Path(json_base_dir)
    tables_base     = Path(tables_base_dir)
    latex_base      = Path(latex_base_dir)

    # ------------------------------------------------------------------
    print("=" * 70)
    print("ETAPA 1: Rodando instâncias para cada configuração")
    print("=" * 70)
    for config_name, build_fn in configs.items():
        print(f"\n>>> {config_name}")
        run_all_instances_for_config(
            config_name=config_name,
            build_fn=build_fn,
            problems_dir=problems_path,
            json_base_dir=json_base_path,
        )

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ETAPA 2: Lendo JSONs e gerando tabelas por configuração")
    print("=" * 70)

    all_rows_by_config = {}

    for config_name in configs:
        json_dir = json_base_path / config_name
        print(f"\n>>> {config_name}")

        rows              = collect_all_metrics(json_dir)
        presolve_rule_rows = collect_presolve_rule_rows(json_dir)
        all_rows_by_config[config_name] = rows

        generate_tables_for_config(
            config_name=config_name,
            rows=rows,
            presolve_rule_rows=presolve_rule_rows,
            tables_base=tables_base,
            latex_base=latex_base,
        )
        print(f"  Tabelas salvas em: {tables_base / config_name}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ETAPA 3: Gerando tabela comparativa entre configurações")
    print("=" * 70)
    generate_comparison_table(
        all_rows_by_config=all_rows_by_config,
        tables_base=tables_base,
        latex_base=latex_base,
    )

    print("\n✅ Processo finalizado.")

if __name__ == "__main__":
    run_experiments_and_generate_tables(
        problems_dir="problems",
        json_base_dir="results_json",
        tables_base_dir="results_tables",
        latex_base_dir="results_latex",
    )
