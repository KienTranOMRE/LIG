"""
run_lamp.py — Run all IG methods + LAMP and compare
====================================================

Usage:
    python run_lamp.py                          # basic run
    python run_lamp.py --viz                    # + heatmaps
    python run_lamp.py --insdel                 # + insertion/deletion
    python run_lamp.py --guided-init            # warm-start LAMP from Guided IG
    python run_lamp.py --viz --insdel --steps 50

Drop this file next to unified_ig.py, utilss.py, lamp.py and run.
"""

import argparse
import json

from lam import (
    get_device,
    standard_ig, idgi, guided_ig, mu_optimized_ig, joint_ig,
    load_image_and_model, _forward_scalar,
    visualize_attributions,
)
from utilss import (
    run_insertion_deletion,
    visualize_step_fidelity,
    visualize_insertion_deletion,
)
from lamp import lamp, mu_lamp, print_decomposition


def run(args):
    device = get_device(force=args.device)

    print("Loading ResNet-50 and image...")
    model, x, baseline, info = load_image_and_model(device, args.min_conf)

    f_x  = _forward_scalar(model, x)
    f_bl = _forward_scalar(model, baseline)
    delta_f = f_x - f_bl
    N = args.steps

    print(f"\nModel : {info['model']}")
    print(f"Source: {info['source']}")
    print(f"Class : {info['target_class']}  conf={info['confidence']:.4f}")
    print(f"f(x)={f_x:.4f}  f(bl)={f_bl:.4f}  Δf={delta_f:.4f}  N={N}\n")

    # ── Run all methods ───────────────────────────────────────────────────
    ig   = standard_ig(model, x, baseline, N)
    idgi_ = idgi(model, x, baseline, N)
    gig  = guided_ig(model, x, baseline, N)
    muopt = mu_optimized_ig(model, x, baseline, N, tau=0.005, n_iter=300)

    G = 16
    init_path = gig.gamma_pts if args.guided_init else None

    joint = joint_ig(
        model, x, baseline, N, G=G,
        n_alternating=2, tau=0.005, mu_iter=300, path_iter=G,
        init_path=init_path,
    )

    # ── LAMP methods ──────────────────────────────────────────────────────
    mulamp = mu_lamp(model, x, baseline, N, tau=0.005, n_iter=300)

    lamp_result = lamp(
        model, x, baseline, N, G=G,
        n_alternating=2, tau=0.005, mu_iter=300, path_iter=G,
        init_path=init_path,
    )

    methods = [ig, idgi_, gig, muopt, joint, mulamp, lamp_result]

    # ── Print Q table ─────────────────────────────────────────────────────
    print(f"\n{'Method':<16} {'Var_ν':>10} {'CV²':>8} {'Q':>8} {'Time':>8}")
    print("─" * 56)
    for m in methods:
        print(f"{m.name:<16} {m.Var_nu:>10.6f} {m.CV2:>8.4f} "
              f"{m.Q:>8.4f} {m.elapsed_s:>7.1f}s")

    # ── MSE_ν decomposition ───────────────────────────────────────────────
    print_decomposition(methods, device)

    # ── Insertion / Deletion ──────────────────────────────────────────────
    if args.insdel:
        run_insertion_deletion(model, x, baseline, methods,
                               n_steps=args.insdel_steps)

    # ── Visualisations ────────────────────────────────────────────────────
    if args.viz:
        visualize_attributions(
            x, methods, info,
            save_path=args.viz_path,
            delta_f=delta_f,
        )
        print(f"✓ Heatmap → {args.viz_path}")

    if args.viz_fidelity:
        fpath = args.viz_path.replace(".png", "_fidelity.png")
        visualize_step_fidelity(methods, save_path=fpath)

    if args.viz_insdel and args.insdel:
        ipath = args.viz_path.replace(".png", "_insdel.png")
        visualize_insertion_deletion(methods, save_path=ipath)

    # ── JSON export ───────────────────────────────────────────────────────
    if args.json:
        results = {
            "image_info": info,
            "model_info": {
                "f_x": f_x, "f_baseline": f_bl,
                "delta_f": delta_f, "N": N, "device": str(device),
            },
            "methods": {m.name: m.to_dict() for m in methods},
        }
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results → {args.json}")

    return methods


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAMP experiment runner")
    parser.add_argument("--steps",        type=int,   default=50)
    parser.add_argument("--device",       type=str,   default=None)
    parser.add_argument("--min-conf",     type=float, default=0.70)
    parser.add_argument("--guided-init",  action="store_true",
                        help="Warm-start LAMP/Joint from Guided IG path")
    parser.add_argument("--viz",          action="store_true")
    parser.add_argument("--viz-path",     type=str,   default="lamp_heatmaps.png")
    parser.add_argument("--viz-fidelity", action="store_true")
    parser.add_argument("--insdel",       action="store_true")
    parser.add_argument("--insdel-steps", type=int,   default=100)
    parser.add_argument("--viz-insdel",   action="store_true")
    parser.add_argument("--json",         type=str,   default=None)
    args = parser.parse_args()

    run(args)