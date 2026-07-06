from pathlib import Path
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for build_dir in ("build", "build_new"):
    build_path = next(
        (root / build_dir
         for root in (Path.cwd(), *Path.cwd().parents)
         if (root / build_dir).is_dir()),
        None,
    )
    if build_path is not None:
        sys.path.insert(0, str(build_path))
        break

import lcbinint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot binary-lens caustics, critical curves, source, and images."
    )
    parser.add_argument("--q", type=float, default=1.0e-3, help="Mass ratio.")
    parser.add_argument("--s", type=float, default=1.0, help="Lens separation.")
    parser.add_argument("--x", type=float, default=0.01, help="Source x position.")
    parser.add_argument("--y", type=float, default=-0.02, help="Source y position.")
    parser.add_argument("--rho", type=float, default=2.0e-3, help="Source radius.")
    parser.add_argument(
        "--n-points", type=int, default=512,
        help="Number of points per caustic/critical-curve branch."
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path for saving the plot. No file is written by default."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plane = lcbinint.image.binary(
        q=args.q,
        s=args.s,
        x=args.x,
        y=args.y,
        rho=args.rho,
        n_points=args.n_points,
    )

    images = plane.image_table()
    print(f"source=({args.x:.6g}, {args.y:.6g}) q={args.q:.6g} s={args.s:.6g}")
    print(f"found {len(images)} point-source images")
    for i, row in enumerate(images):
        print(
            f"  image {i}: x={row['x']:.8f} y={row['y']:.8f} "
            f"mu={row['magnification']:.8f} parity={row['parity']:+d}"
        )

    ax = plane.plot(legend=True)
    ax.set_title("Binary-lens image-plane geometry")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(args.output, dpi=160)
        print(f"saved {args.output}")
    plt.close(ax.figure)


if __name__ == "__main__":
    main()
