from branch_and_bound import *
from cuts import *
from presolve import *
from primal_heuristics import *
from branching import *
from node_selection import *


model_path = "problems/stress_17.mps"

bb = BranchAndBound(
    model_path=model_path,
    strategy=BestFirst(),
    direction=BranchDirection.CEIL_FIRST,
    branch_rule=StrongBranchingRule(max_candidates=8, score_mode="product"),
    presolver=Presolver([BoundTightening(),
                RowFeasibilityAndRedundancy(),
                SubstituteFixedVariables(),
                ImplicitFreeVariableDetection(),
                SingletonRowReduction(),
                EuclideanReduction(),
                CoefficientStrengthening(),
                ChvatalGomoryStrengthening(),
                ParallelRows(),
                ],
                verbose=True),
    heuristics=[
        RoundingHeuristic(only_root=False,   probability=1.0),
        FixAndPropagate(only_root=False,      probability=0.3),
        RENS(only_root=True,                  probability=1.0),
        RINS(only_root=False,                 probability=0.3),
        LocalSearch(only_root=False,          probability=0.3),
        LocalBranching(only_root=False,       probability=0.3),
        LNS(FractionalGuidedDestroy(),        probability=0.3),
        LNS(RandomDestroy(),                  probability=0.2),
        LNS(ConstraintBasedDestroy(),         probability=0.2),
        LNS(ObjectiveBasedDestroy(),          probability=0.2),
        ALNS(probability=0.3),                                   
        ALNS(                                                     
            acceptance=ALNSAcceptance.SIMULATED_ANNEAL,
            probability=0.2,
        ),
        FractionalDiving(only_root=False,     probability=0.3),
        GuidedDiving(only_root=False,         probability=0.3),
        # CoefficientDiving(only_root=False,    probability=0.3),
        LockDiving(only_root=False,           probability=0.3),
        FeasibilityPump(only_root=False,      probability=0.3),
        FeasibilityJump(only_root=False,      probability=0.3),
    ],
    cutting_planes=[
    # MixedIntegerGomoryCut(only_root=True, max_cuts=20),
    # SimpleMIR(only_root=True, max_cuts=20),
    LiftingExtending(only_root=True, max_cuts=20),
    # CoverCut(only_root=True, max_cuts=10),
    ],
    verbose=True,
)
result = bb.solve()

if result["status"] == "optimal":
    print(f"  Valor total: {result['obj_value']:.4f}")

bb.export_stats("results_stress_17.json")
