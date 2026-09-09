"""
compiler/operators/gpl_operator_decorator.py
=============================================
Decorator that auto-generates an executable formula_line for every GPL operator.

PROBLEM SOLVED:
  Previously, every operator manually built a human-readable formula string like:
    MEASURE(SUM(payload_capacity_kg) FROM logistics_vehicles_MANUAL_dimension)
  This looks nice but is NOT valid Python and cannot be exec()'d at the customer site.

SOLUTION:
  The @gpl_operator decorator wraps each operator function. When the operator is called,
  the decorator captures the exact function name and arguments, then builds formula_line
  as a proper Python-callable string:
    MEASURE('logistics_vehicles_MANUAL_dimension', 'SUM', 'payload_capacity_kg', None)
  This is valid Python that can be exec()'d directly at the customer runtime against
  their real data using the same operator functions in the namespace.

USAGE:
  @gpl_operator
  def MEASURE(atom_canonical_id, agg, column, filters=None):
      # ... computation only ...
      return {"oracle_value": value}

  The decorator automatically injects formula_line into the returned dict.
  Operators do NOT need to build formula strings manually anymore.

GUARANTEE:
  formula_line is always built from the ACTUAL arguments the operator was called with.
  It is impossible for formula_line to drift from the actual computation.
  New operators get correct formula_line for free just by adding @gpl_operator.

CUSTOMER RUNTIME:
  At the customer site, the runtime does:
    ns = build_namespace(csv_root)   # registers all operator functions
    exec(f"__r__ = {formula_line}", ns)
    oracle_value = ns["__r__"].get("oracle_value")

  This works because formula_line is a valid Python call to a registered operator.
"""

import inspect
import functools
from typing import Any, Callable, Dict


def gpl_operator(fn: Callable) -> Callable:
    """
    Decorator for GPL operator functions.

    Wraps the operator to automatically build and inject an executable
    formula_line into the returned dict after the operator runs.

    The formula_line is built by capturing:
      - fn.__name__: the operator name (e.g. "MEASURE")
      - The bound arguments at call time (positional + keyword, with defaults filled in)

    Args are serialized using repr() so the resulting string is valid Python:
      - Strings become 'quoted'
      - Lists become ['item1', 'item2']
      - Dicts become [{'field': 'status', 'op': '=', 'value': 'pending'}]
      - None stays None

    The resulting formula_line is always a valid Python expression that can be
    exec()'d in a namespace where the operator functions are registered.
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Run the operator — get the result dict
        result = fn(*args, **kwargs)

        # Bind all arguments (positional + keyword) to parameter names,
        # filling in defaults for any parameters not explicitly passed
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Serialize each argument to its repr so the string is valid Python
        arg_parts = [repr(v) for v in bound.arguments.values()]

        # Build the executable formula string
        formula_line = f"{fn.__name__}({', '.join(arg_parts)})"

        # Inject into result — overwrite any manually built formula_line
        if isinstance(result, dict):
            result["formula_line"] = formula_line

        return result

    return wrapper
