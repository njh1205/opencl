#!/usr/bin/env python3
"""registry_check: prove the operator library is reachable on BOTH paths.

A kernel that exists in kernels/ but is missing from the host registry is
source-JIT-only: it works on the dev host and silently does not exist on the
PL, because load_xclbin() can only acquire names it asks for. That is exactly
how the eight per-symbol datapath kernels were lost (they were reachable only
from a private mini-runtime inside runtime/datapath_main.cpp).

So this gate cross-compares four lists that must agree:

  A  kernels/*.cl            every `__kernel void NAME` that is compiled
  B  runtime/hpocl_runtime.cpp::kKernelNames   every name the host can acquire
  C  opencl/build/build_xclbin.sh  every name the PL link actually exports
  D  runtime/*.cpp, ref/*.cpp     at least one call site per kernel

A==B==C is required. D is required too: a kernel nobody calls is dead weight in
the xclbin (it still costs area).

It also checks E: no two kernel sources may define the same macro or the same
non-static function. Runtime::build() concatenates every .cl it is handed into a
SINGLE clCreateProgramWithSource, so a shared name is not a style problem -- it
is a silent resize of somebody else's array. That is not hypothetical: MAXD was
8 in decomp.cl (private R/P scratch) and 128 in threshold_cf.cl (LLR distance
candidates), and a bench that loaded both would have given threshold_cf an 8
entry d0[]/d1[] and overflowed it 16x. The two were renamed apart; this check is
what keeps them apart.

Pure text analysis -- no OpenCL device, no build, so it runs anywhere.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def kernel_dir():
    d = os.path.join(ROOT, "kernels")
    return sorted(f for f in os.listdir(d) if f.endswith(".cl"))


def main():
    # A: what the device sources define
    a = {}
    for f in kernel_dir():
        for name in re.findall(r"__kernel\s+void\s+([A-Za-z_]\w*)", read("kernels", f)):
            a.setdefault(name, f)
    dup = [n for n in a if sum(read("kernels", f).count("__kernel void %s(" % n)
                              for f in kernel_dir()) > 1]
    if dup:
        print("[registry_check] FAIL kernel defined twice: %s" % ", ".join(sorted(dup)))
        return 1

    # B: what the host registry asks for
    rt = read("runtime", "hpocl_runtime.cpp")
    m = re.search(r"kKernelNames\[\]\s*=\s*\{(.*?)\};", rt, re.S)
    if not m:
        print("[registry_check] FAIL cannot find kKernelNames[] in hpocl_runtime.cpp")
        return 1
    b = set(re.findall(r'"([A-Za-z_]\w*)"', m.group(1)))

    # C: what the PL link exports
    sh = read("opencl", "build", "build_xclbin.sh")
    c = set(re.findall(r"-k\s+([A-Za-z_]\w*)", sh))

    # D: at least one host call site. A name counts as called when it appears as
    # a string literal anywhere in runtime/ or ref/ OUTSIDE the kKernelNames
    # registry block -- that covers both the direct kernel("gmd_k") form and the
    # table-driven kTfKernelNames[i] form the transformer predictor uses, and it
    # does not let "registered" masquerade as "used".
    reg_block = m.group(1)
    callers = {}
    for sub in ("runtime", "ref"):
        d = os.path.join(ROOT, sub)
        for f in sorted(os.listdir(d)):
            if not f.endswith((".cpp", ".h")):
                continue
            txt = read(sub, f).replace(reg_block, "")
            for name in re.findall(r'"([A-Za-z_]\w*)"', txt):
                if name in a:
                    callers.setdefault(name, "%s/%s" % (sub, f))

    # E: macro / non-static function name collisions across kernel sources
    macros, funcs = {}, {}
    fn_re = re.compile(r"^(?!static)(?:inline\s+)?(?:void|int|float|float2|double|"
                       r"real|uchar|uint|long|size_t)\s+([A-Za-z_]\w*)\s*\(", re.M)
    for f in kernel_dir() + sorted(x for x in os.listdir(os.path.join(ROOT, "kernels"))
                                   if x.endswith(".clh")):
        txt = read("kernels", f)
        for name in re.findall(r"^\s*#define\s+([A-Za-z_]\w*)", txt, re.M):
            macros.setdefault(name, set()).add(f)
        for name in fn_re.findall(txt):
            if name not in a:                 # kernels are entry points, not helpers
                funcs.setdefault(name, set()).add(f)
    dup_macros = sorted(n for n, fs in macros.items() if len(fs) > 1)
    dup_funcs = sorted(n for n, fs in funcs.items() if len(fs) > 1)
    if dup_macros:
        print("[registry_check] FAIL macro defined in more than one kernel source "
              "(Runtime::build concatenates them into one program):")
        for n in dup_macros:
            print("                 %-18s %s" % (n, ", ".join(sorted(macros[n]))))
        bad_macro = 1
    else:
        bad_macro = 0
    if dup_funcs:
        print("[registry_check] FAIL non-static helper defined in more than one "
              "kernel source:")
        for n in dup_funcs:
            print("                 %-18s %s" % (n, ", ".join(sorted(funcs[n]))))
        bad_macro = 1

    bad = bad_macro
    for label, s in (("registry (B)", b), ("xclbin (C)", c)):
        ghost = sorted(s - set(a))
        if ghost:
            print("[registry_check] FAIL %s names kernels that no .cl defines: %s"
                  % (label, ", ".join(ghost)))
            bad += 1
    for label, s in (("registry (B)", b), ("xclbin (C)", c)):
        miss = sorted(set(a) - s)
        if miss:
            print("[registry_check] FAIL %s is missing %d kernel(s): %s"
                  % (label, len(miss), ", ".join(miss)))
            bad += 1
    nocall = sorted(n for n in a if n not in callers)
    if nocall:
        print("[registry_check] FAIL %d kernel(s) have no host call site (dead "
              "weight in the xclbin): %s" % (len(nocall), ", ".join(nocall)))
        bad += 1
    if bad:
        return 1

    # also check the two acquisition paths really share one name list
    if rt.count("kKernelNames") < 3:
        print("[registry_check] FAIL kKernelNames is not used by both build() and "
              "load_xclbin() -- the two paths have drifted apart again")
        return 1
    if "acquire_kernels()" not in rt:
        print("[registry_check] FAIL acquire_kernels() is gone")
        return 1

    print("[registry_check] OK: %d kernels in %d sources; registry, xclbin and "
          "call sites all agree; %d macros and %d helpers collision-free"
          % (len(a), len(kernel_dir()), len(macros), len(funcs)))
    for name in sorted(a):
        print("    %-20s %-22s %s" % (name, a[name], callers[name]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
