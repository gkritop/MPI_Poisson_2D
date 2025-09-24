#!/usr/bin/env python3
import argparse, csv, math
import matplotlib.pyplot as plt

def read_csv(path):
    rows = []
    with open(path, newline='') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append({
                "mode": r["mode"],
                "nprocs": int(r["nprocs"]),
                "px": int(r["px"]), "py": int(r["py"]),
                "nxg": int(r["nxg"]), "nyg": int(r["nyg"]),
                "nxl": int(r["nxl"]), "nyl": int(r["nyl"]),
                "iters": int(r["iters"]),
                "time_s": float(r["time_s"]),
                "residual": float(r["residual"]),
            })
    return rows

def plot_strong(ax_time, ax_speedup, rows):
    rows = sorted(rows, key=lambda r: r["nprocs"])

    procs = [r["nprocs"] for r in rows]
    times = [r["time_s"] for r in rows]
    t1 = times[0]

    speedup = [t1/t for t in times]
    ideal = procs

    ax_time.plot(procs, times, marker="o", label="Measured")
    ax_time.set_xscale("log", base=2)
    ax_time.set_xlabel("Ranks (log2)")
    ax_time.set_ylabel("Time (s)")
    ax_time.set_title("Strong scaling: time vs ranks")
    ax_time.grid(True, which="both")

    ax_speedup.plot(procs, speedup, marker="o", label="Measured")
    ax_speedup.plot(procs, ideal, linestyle="--", label="Ideal")
    ax_speedup.set_xscale("log", base=2)
    ax_speedup.set_xlabel("Ranks (log2)")
    ax_speedup.set_ylabel("Speedup (×)")
    ax_speedup.set_title("Strong scaling: speedup vs ideal")
    ax_speedup.legend()
    ax_speedup.grid(True, which="both")

def plot_weak(ax_time, ax_speedup, rows):
    rows = sorted(rows, key=lambda r: r["nprocs"])

    procs = [r["nprocs"] for r in rows]
    times = [r["time_s"] for r in rows]

    # Time vs ranks
    ax_time.plot(procs, times, marker="o", label="Measured")
    ax_time.set_xscale("log", base=2)
    ax_time.set_xlabel("Ranks (log2)")
    ax_time.set_ylabel("Time (s)")
    ax_time.set_title("Weak scaling: time vs ranks")
    ax_time.grid(True, which="both")

    # Speedup vs ideal (weak scaling definition)
    t1 = times[0]
    speedup = [(p * t1) / t for p, t in zip(procs, times)]
    ideal = procs

    ax_speedup.plot(procs, speedup, marker="o", label="Measured")
    ax_speedup.plot(procs, ideal, linestyle="--", label="Ideal")
    ax_speedup.set_xscale("log", base=2)
    ax_speedup.set_xlabel("Ranks (log2)")
    ax_speedup.set_ylabel("Speedup (×)")
    ax_speedup.set_title("Weak scaling: speedup vs ideal")
    ax_speedup.legend()
    ax_speedup.grid(True, which="both")

def main():
    p = argparse.ArgumentParser(description="Plot strong/weak scaling from CSV exports")
    
    p.add_argument("strong_csv", help="results/strong.csv")
    p.add_argument("weak_csv", nargs="?", help="results/weak.csv (optional)")
    p.add_argument("--save", help="Path to save PNG (optional). If omitted, shows interactively.")
    
    args = p.parse_args()

    strong = read_csv(args.strong_csv)
    weak = read_csv(args.weak_csv) if args.weak_csv else None

    # Strong
    fig1 = plt.figure(); ax1 = fig1.gca()
    fig2 = plt.figure(); ax2 = fig2.gca()
    plot_strong(ax1, ax2, strong)

    # Weak
    if weak:
        fig3 = plt.figure(); ax3 = fig3.gca()
        fig4 = plt.figure(); ax4 = fig4.gca()
        
        plot_weak(ax3, ax4, weak)

    if args.save:
        paths = []

        for i, num in enumerate(plt.get_fignums(), start=1):
            out = args.save
            if len(plt.get_fignums()) > 1:
                stem, ext = (args.save.rsplit('.', 1) + ["png"])[:2]
                out = f"{stem}-{i}.{ext}"
            
            plt.figure(num).savefig(out, bbox_inches="tight", dpi=150)
            paths.append(out)

        print("Saved:", ", ".join(paths))
    else:
        plt.show()

if __name__ == "__main__":
    main()
