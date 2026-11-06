"""
Speed Limit - router tests (exercise SpeedLimit10b).

Usage:  python test_router.py

The router (Software 1.0) decides: "reduce the speed limit because of the weather?" -> yes / no.
Inputs (integers): light in millilux, water on the road in micrometers, temperature in degrees Celsius.
Rule (SPEC1, SPEC2):   yes  if  light < 500   OR   (water > 1000 AND temperature < 0)

Four test cases, chosen so that they find as many bugs as possible:
  * every condition is tested exactly at its boundary (off-by-one bugs: < vs <=),
  * both conditions of the AND are tested separately (a missing or wrong AND),
  * the OR between "dark" and "ice" is tested (dark alone must be enough).
Negative testing (invalid input such as text or negative water) is out of scope for this exercise.

The second part of the script checks the tests themselves: it creates deliberately broken copies of the
router (mutants) and shows that at least one of the four tests fails for each of them.
"""
import sys
from speedlimit_core import router_uses_nn

# ID, light (millilux), water (micrometers), temperature (C), expected output, what the test checks
TESTS = [
    ("T1", 499,    0,   20, "yes", "darkness alone is enough; 499 is the last dark value (light < 500)"),
    ("T2", 500, 1000,   -1, "no",  "500 is not dark; 1000 micrometers is not 'more than 1000' (water boundary)"),
    ("T3", 500, 1001,    0, "no",  "water is above 1000 but 0 degrees is not below zero (temperature boundary)"),
    ("T4", 500, 1001,   -1, "yes", "ice risk: water above 1000 AND temperature below zero"),
]


def as_text(flag):
    return "yes" if flag else "no"


# ---------------------------------------------------------------- part 1: run the tests
print("PART 1: router tests\n")
print(f"{'ID':4s} {'light':>6s} {'water':>6s} {'temp':>5s}  {'expected':8s} {'actual':8s} result")
failed = 0
for tid, light, water, temp, expected, _why in TESTS:
    actual = as_text(router_uses_nn(light, water, temp))
    ok = actual == expected
    failed += not ok
    print(f"{tid:4s} {light:6d} {water:6d} {temp:5d}  {expected:8s} {actual:8s} {'PASS' if ok else 'FAIL'}")
print()
for tid, *_rest, why in TESTS:
    print(f"{tid}: {why}")

# ---------------------------------------------------------------- part 2: do the tests catch bugs?
MUTANTS = {
    "dark uses <= 500 instead of < 500":        lambda l, w, t: l <= 500 or (w > 1000 and t < 0),
    "dark threshold 499 instead of 500":        lambda l, w, t: l < 499 or (w > 1000 and t < 0),
    "dark threshold 450 instead of 500":        lambda l, w, t: l < 450 or (w > 1000 and t < 0),
    "water uses >= 1000 instead of > 1000":     lambda l, w, t: l < 500 or (w >= 1000 and t < 0),
    "water threshold 1001 instead of 1000":     lambda l, w, t: l < 500 or (w > 1001 and t < 0),
    "temperature uses <= 0 instead of < 0":     lambda l, w, t: l < 500 or (w > 1000 and t <= 0),
    "temperature threshold -1 instead of 0":    lambda l, w, t: l < 500 or (w > 1000 and t < -1),
    "ice uses OR instead of AND":               lambda l, w, t: l < 500 or (w > 1000 or t < 0),
    "ice ignores water (only temperature)":     lambda l, w, t: l < 500 or t < 0,
    "ice ignores temperature (only water)":     lambda l, w, t: l < 500 or w > 1000,
    "dark and ice combined with AND, not OR":   lambda l, w, t: l < 500 and (w > 1000 and t < 0),
    "darkness ignored (ice only)":              lambda l, w, t: w > 1000 and t < 0,
    "always yes":                               lambda l, w, t: True,
    "always no":                                lambda l, w, t: False,
}

print("\n\nPART 2: do the four tests catch bugs? (each broken router should fail at least one test)\n")
print(f"{'broken version of the router':44s} failing tests")
survivors = []
for name, fn in MUTANTS.items():
    caught = [tid for tid, l, w, t, exp, _ in TESTS if as_text(fn(l, w, t)) != exp]
    if not caught:
        survivors.append(name)
    print(f"{name:44s} {', '.join(caught) if caught else '-- NOT DETECTED --'}")

print(f"\nBugs detected: {len(MUTANTS) - len(survivors)} of {len(MUTANTS)}")
only_one = {tid: [n for n, fn in MUTANTS.items()
                  if [x for x, l, w, t, e, _ in TESTS if as_text(fn(l, w, t)) != e] == [tid]] for tid, *_ in TESTS}
print("Bugs found ONLY by one specific test (so the test cannot be removed):")
for tid, names in only_one.items():
    print(f"  {tid}: {len(names)}" + (f"  e.g. {names[0]}" if names else ""))

sys.exit(1 if failed or survivors else 0)
