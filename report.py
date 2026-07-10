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
JSON_DIR = Path("results_json")
TABLES_DIR = Path("results_tables")
LATEX_DIR = Path("results_latex")

JSON_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR.mkdir(parents=True, exist_ok=True)

def build_solver(model_path: str) -> BranchAndBound:
    bb = BranchAndBound(
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
            verbose=True,
        ),
        heuristics=[
            RoundingHeuristic(only_root=False, probability=1.0),
            FixAndPropagate(only_root=False, probability=0.3),
            RENS(only_root=True, probability=1.0),
            RINS(only_root=False, probability=0.3),
            LocalSearch(only_root=False, probability=0.3),
            LocalBranching(only_root=False, probability=0.3),
            LNS(FractionalGuidedDestroy(), probability=0.3),
            ALNS(
                acceptance=ALNSAcceptance.SIMULATED_ANNEAL,
                probability=0.2,
            ),
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
    return bb

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
        if math.isinf(x):
            return ""
        if math.isnan(x):
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
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
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

def run_all_instances(problems_dir="problems", json_dir="results_json"):
    problems_path = Path(problems_dir)
    json_path = Path(json_dir)
    json_path.mkdir(parents=True, exist_ok=True)

    mps_files = sorted(problems_path.glob("*.mps"))
    print(f"Encontradas {len(mps_files)} instâncias em {problems_path}/")

    for mps_file in mps_files:
        print(f"Rodando: {mps_file.name}")
        try:
            bb = build_solver(str(mps_file))
            bb.solve()

            out_json = json_path / f"{mps_file.stem}.json"
            bb.export_stats(str(out_json))
            print(f"  JSON salvo em: {out_json}")

        except Exception as e:
            print(f"  ERRO em {mps_file.name}: {e}")

def extract_metrics_from_json(json_file: Path) -> dict:
    with open(json_file, "r", encoding="utf-8") as f:
        stats = json.load(f)

    instance_name = json_file.stem

    n_vars = safe_get(stats, "instance", "n_vars")
    n_constraints = safe_get(stats, "instance", "n_constraints")

    status = safe_get(stats, "final", "status")
    obj_value = safe_get(stats, "final", "obj_value")

    root_lp = safe_get(stats, "root_lp", "lp_value")
    root_lp_after_cuts = safe_get(stats, "root_lp", "lp_value_after_cuts")

    gap_root = None
    gap_root_after_cuts = None

    if status == "optimal" and obj_value is not None:
        gap_root = compute_root_gap(obj_value, root_lp)
        gap_root_after_cuts = compute_root_gap(obj_value, root_lp_after_cuts)

    runtime_total = safe_get(stats, "final", "runtime_total")
    runtime_lp = safe_get(stats, "final", "runtime_lp")
    runtime_heuristics = safe_get(stats, "final", "runtime_heuristics")
    runtime_cuts = safe_get(stats, "final", "runtime_cuts")

    nodes_explored = safe_get(stats, "tree", "nodes_explored")
    nodes_pruned = safe_get(stats, "tree", "nodes_pruned")
    pruned_by_infeasibility = safe_get(stats, "tree", "pruned_by_infeasibility")
    pruned_by_bound = safe_get(stats, "tree", "pruned_by_bound")
    max_depth = safe_get(stats, "tree", "max_depth")
    max_open_nodes = safe_get(stats, "tree", "max_open_nodes")

    n_incumbents = len(stats.get("incumbents", []))

    total_heur_calls = sum_heuristic_metric(stats, "calls")
    total_heur_improvements = sum_heuristic_metric(stats, "improvements")
    total_heur_solutions = sum_heuristic_metric(stats, "solutions_found")

    total_cuts_added = sum_cut_metric(stats, "cuts_added")

    presolve_summary = stats.get("presolve", {}).get("summary", {})
    presolve_runtime = presolve_summary.get("total_runtime")
    presolve_bounds = presolve_summary.get("total_bounds_tightened")
    presolve_rows_removed = presolve_summary.get("total_rows_removed")
    presolve_vars_fixed = presolve_summary.get("total_vars_fixed")
    presolve_coeff_changed = presolve_summary.get("total_coefficients_changed")
    presolve_n_rules = presolve_summary.get("n_rules")

    pct_rows_removed = None
    pct_vars_fixed = None
    pct_presolve_time = None

    if n_constraints not in (None, 0) and presolve_rows_removed is not None:
        pct_rows_removed = 100.0 * presolve_rows_removed / n_constraints

    if n_vars not in (None, 0) and presolve_vars_fixed is not None:
        pct_vars_fixed = 100.0 * presolve_vars_fixed / n_vars

    if runtime_total not in (None, 0) and presolve_runtime is not None:
        pct_presolve_time = 100.0 * presolve_runtime / runtime_total

    return {
        "instance": instance_name,
        "status": status,
        "obj_value": round_or_none(obj_value),
        "root_lp": round_or_none(root_lp),
        "root_lp_after_cuts": round_or_none(root_lp_after_cuts),
        "gap_root": round_or_none(gap_root),
        "gap_root_after_cuts": round_or_none(gap_root_after_cuts),
        "runtime_total": round_or_none(runtime_total),
        "runtime_lp": round_or_none(runtime_lp),
        "runtime_heuristics": round_or_none(runtime_heuristics),
        "runtime_cuts": round_or_none(runtime_cuts),
        "nodes_explored": nodes_explored,
        "nodes_pruned": nodes_pruned,
        "pruned_by_infeasibility": pruned_by_infeasibility,
        "pruned_by_bound": pruned_by_bound,
        "max_depth": max_depth,
        "max_open_nodes": max_open_nodes,
        "n_incumbents": n_incumbents,
        "total_heur_calls": total_heur_calls,
        "total_heur_improvements": total_heur_improvements,
        "total_heur_solutions": total_heur_solutions,
        "total_cuts_added": total_cuts_added,
        "presolve_n_rules": presolve_n_rules,
        "presolve_bounds_tightened": presolve_bounds,
        "presolve_rows_removed": presolve_rows_removed,
        "presolve_vars_fixed": presolve_vars_fixed,
        "presolve_coefficients_changed": presolve_coeff_changed,
        "presolve_runtime": round_or_none(presolve_runtime),
        "pct_rows_removed": round_or_none(pct_rows_removed),
        "pct_vars_fixed": round_or_none(pct_vars_fixed),
        "pct_presolve_time": round_or_none(pct_presolve_time),
    }

def collect_all_metrics(json_dir="results_json"):
    json_path = Path(json_dir)
    json_files = sorted(json_path.glob("*.json"))

    rows = []
    for jf in json_files:
        try:
            rows.append(extract_metrics_from_json(jf))
        except Exception as e:
            print(f"Erro lendo {jf.name}: {e}")

    return rows

def collect_presolve_rule_rows(json_dir="results_json"):
    json_path = Path(json_dir)
    json_files = sorted(json_path.glob("*.json"))

    rows = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                stats = json.load(f)

            instance_name = jf.stem
            rules = safe_get(stats, "presolve", "rules", default=[])

            for rule_data in rules:
                rows.append({
                    "instance": instance_name,
                    "rule": rule_data.get("rule"),
                    "bounds_tightened": rule_data.get("bounds_tightened"),
                    "rows_removed": rule_data.get("rows_removed"),
                    "vars_fixed": rule_data.get("vars_fixed"),
                    "coefficients_changed": rule_data.get("coefficients_changed"),
                    "runtime": round_or_none(rule_data.get("runtime")),
                    "active": "yes" if presolve_rule_is_active(rule_data) else "no",
                    "infeasible": "yes" if rule_data.get("infeasible", False) else "no",
                })
        except Exception as e:
            print(f"Erro lendo presolve de {jf.name}: {e}")

    return rows

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
                if k == "instance" or k == "status" or k == "rule" or k == "active" or k == "infeasible":
                    vals.append(latex_escape(v))
                else:
                    vals.append(latex_escape(fmt_latex_value(v)))
            f.write(" & ".join(vals) + " \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

def generate_tables(rows, presolve_rule_rows, tables_dir="results_tables", latex_dir="results_latex"):
    tables_path = Path(tables_dir)
    latex_path = Path(latex_dir)

    tables_path.mkdir(parents=True, exist_ok=True)
    latex_path.mkdir(parents=True, exist_ok=True)

    table1_fields = [
        "instance",
        "status",
        "obj_value",
        "root_lp",
        "root_lp_after_cuts",
        "gap_root",
        "gap_root_after_cuts",
    ]
    table1_rows = [{k: row.get(k) for k in table1_fields} for row in rows]
    write_csv(tables_path / "table_1_resultado_relaxacao.csv", table1_fields, table1_rows)
    write_latex_table(
        latex_path / "table_1_resultado_relaxacao.tex",
        "Resultado final e qualidade da relaxação.",
        "tab:resultado_relaxacao",
        table1_fields,
        table1_rows,
    )

    table2_fields = [
        "instance",
        "nodes_explored",
        "nodes_pruned",
        "pruned_by_infeasibility",
        "pruned_by_bound",
        "max_depth",
        "max_open_nodes",
    ]
    table2_rows = [{k: row.get(k) for k in table2_fields} for row in rows]
    write_csv(tables_path / "table_2_arvore.csv", table2_fields, table2_rows)
    write_latex_table(
        latex_path / "table_2_arvore.tex",
        "Tamanho e esforço da árvore Branch-and-Bound.",
        "tab:arvore",
        table2_fields,
        table2_rows,
    )

    table3_fields = [
        "instance",
        "runtime_total",
        "runtime_lp",
        "runtime_heuristics",
        "runtime_cuts",
    ]
    table3_rows = [{k: row.get(k) for k in table3_fields} for row in rows]
    write_csv(tables_path / "table_3_tempos.csv", table3_fields, table3_rows)
    write_latex_table(
        latex_path / "table_3_tempos.tex",
        "Tempos de execução por componente.",
        "tab:tempos",
        table3_fields,
        table3_rows,
    )

    table4_fields = [
        "instance",
        "n_incumbents",
        "total_heur_calls",
        "total_heur_improvements",
        "total_heur_solutions",
        "total_cuts_added",
    ]
    table4_rows = [{k: row.get(k) for k in table4_fields} for row in rows]
    write_csv(tables_path / "table_4_heuristicas_cortes.csv", table4_fields, table4_rows)
    write_latex_table(
        latex_path / "table_4_heuristicas_cortes.tex",
        "Resumo de heurísticas e cortes.",
        "tab:heuristicas_cortes",
        table4_fields,
        table4_rows,
    )

    tablep1_fields = [
        "instance",
        "presolve_n_rules",
        "presolve_bounds_tightened",
        "presolve_rows_removed",
        "presolve_vars_fixed",
        "presolve_coefficients_changed",
        "presolve_runtime",
        "pct_rows_removed",
        "pct_vars_fixed",
        "pct_presolve_time",
    ]
    tablep1_rows = [{k: row.get(k) for k in tablep1_fields} for row in rows]
    write_csv(tables_path / "table_p1_presolve_summary.csv", tablep1_fields, tablep1_rows)
    write_latex_table(
        latex_path / "table_p1_presolve_summary.tex",
        "Resumo do presolve por instância.",
        "tab:presolve_summary",
        tablep1_fields,
        tablep1_rows,
    )

    tablep2_fields = [
        "instance",
        "rule",
        "bounds_tightened",
        "rows_removed",
        "vars_fixed",
        "coefficients_changed",
        "runtime",
        "active",
        "infeasible",
    ]
    tablep2_rows = [{k: row.get(k) for k in tablep2_fields} for row in presolve_rule_rows]
    write_csv(tables_path / "table_p2_presolve_rules.csv", tablep2_fields, tablep2_rows)
    write_latex_table(
        latex_path / "table_p2_presolve_rules.tex",
        "Desempenho das regras de presolve por instância.",
        "tab:presolve_rules",
        tablep2_fields,
        tablep2_rows,
    )

    print(f"Tabelas CSV salvas em: {tables_path}")
    print(f"Tabelas LaTeX salvas em: {latex_path}")

def run_experiments_and_generate_tables(
    problems_dir="problems",
    json_dir="results_json",
    tables_dir="results_tables",
    latex_dir="results_latex",
):
    print("=" * 70)
    print("ETAPA 1: Rodando instâncias e gerando JSONs")
    print("=" * 70)
    run_all_instances(problems_dir=problems_dir, json_dir=json_dir)

    print("\n" + "=" * 70)
    print("ETAPA 2: Lendo JSONs e extraindo métricas")
    print("=" * 70)
    rows = collect_all_metrics(json_dir=json_dir)
    presolve_rule_rows = collect_presolve_rule_rows(json_dir=json_dir)

    print("\n" + "=" * 70)
    print("ETAPA 3: Gerando tabelas CSV e LaTeX")
    print("=" * 70)
    generate_tables(
        rows,
        presolve_rule_rows,
        tables_dir=tables_dir,
        latex_dir=latex_dir,
    )

    print("\nProcesso finalizado.")

if __name__ == "__main__":
    run_experiments_and_generate_tables(
        problems_dir="problems",
        json_dir="results_json",
        tables_dir="results_tables",
        latex_dir="results_latex",
    )