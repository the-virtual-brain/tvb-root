#!/usr/bin/env bash
# ralph_add_model_codegen.sh — Iterative AI-assisted addition of codegen
# attributes to TVB model classes using opencode + zai/glm-5.1.
#
# Usage:
#   cd tvb_library
#   bash tvb/simulator/backend/ralph_add_model_codegen.sh [model_index]
#
# With no argument, processes ALL models sequentially.
# With an index (0-based), processes only that one model.
#
# Each iteration:
#   1. Sends a prompt to opencode with the model file and an exemplar
#   2. Validates that the model class gained the required attributes
#   3. Runs the existing test suite to check for regressions
#   4. Commits on success, retries (up to 3) on failure

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
TEST_FILE="$REPO_ROOT/tvb/tests/library/simulator/backend/test_nb_hybrid.py"

# Provider/model for opencode
PROVIDER="zai/glm-5.1"

# Exemplar: a model that already has all attrs (Generic2dOscillator is clean + simple)
EXEMPLAR_FILE="$REPO_ROOT/tvb/simulator/models/oscillator.py"
EXEMPLAR_CLASS="Generic2dOscillator"

# ---- Model list: (file_relative_to_tvb_library, class_name, notes) ----
# Order: easy → hard.  Each entry is "relative_path:ClassName:hint"
MODELS=(
  "tvb/simulator/models/oscillator.py:SupHopf:2 svars (x,y); complex-valued Hopf; has dfun_helpers already; cvar=[0,1]; coupling_terms should match cvar names from dfun"
  "tvb/simulator/models/oscillator.py:Kuramoto:1 svar (theta); sin-based coupling in dfun; cvar=[0]"
  "tvb/simulator/models/epileptor.py:Epileptor2D:2 svars (x1,z); reduced seizure model; ModelNumbaDfun base; cvar=[0]"
  "tvb/simulator/models/hopfield.py:Hopfield:2 svars (x,theta); simple attractor dynamics; cvar=[0]"
  "tvb/simulator/models/infinite_theta.py:CoombesByrne2D:2 svars; theta-neuron OA reduction"
  "tvb/simulator/models/larter_breakspear.py:LarterBreakspear:3 svars (V,W,Z); tanh nonlinearities; cvar=[0]"
  "tvb/simulator/models/infinite_theta.py:CoombesByrne:4 svars; theta-neuron 4D"
  "tvb/simulator/models/infinite_theta.py:GastSchmidtKnosche_SD:4 svars; theta-neuron variant"
  "tvb/simulator/models/infinite_theta.py:GastSchmidtKnosche_SF:4 svars; theta-neuron variant"
  "tvb/simulator/models/epileptorcodim3.py:EpileptorCodim3:3 svars; ModelNumbaDfun base"
  "tvb/simulator/models/epileptorcodim3.py:EpileptorCodim3SlowMod:5 svars; ModelNumbaDfun base; extends EpileptorCodim3"
  "tvb/simulator/models/wong_wang_exc_inh.py:ReducedWongWangExcInh:2 svars (S_e,S_i); currently uses guvectorize _numba_dfun; need to add codegen attrs alongside existing code"
  "tvb/simulator/models/epileptor_rs.py:EpileptorRestingState:8 svars; large but scalar ops only"
  "tvb/simulator/models/infinite_theta.py:DumontGutkin:8 svars; complex theta-neuron"
  "tvb/simulator/models/jansen_rit.py:ZetterbergJansen:12 svars; 12D version of JansenRit"
  "tvb/simulator/models/zerlaut.py:ZerlautAdaptationFirstOrder:5 svars; adaptive EIF; complex transfer function"
  "tvb/simulator/models/zerlaut.py:ZerlautAdaptationSecondOrder:8 svars; extends first-order Zerlaut"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---- Validation function ----
validate_model() {
    local file="$1" class="$2"
    "$VENV_PYTHON" -c "
import sys, importlib, os
sys.path.insert(0, '$REPO_ROOT')
# Derive module path from file path
rel = os.path.relpath('$REPO_ROOT/$file', '$REPO_ROOT')
mod_path = rel.replace('/', '.').replace('.py', '')
mod = importlib.import_module(mod_path)
cls = getattr(mod, '$class')
m = cls()

errors = []
if not hasattr(m, 'coupling_terms') or not m.coupling_terms:
    errors.append('missing coupling_terms')
if not hasattr(m, 'parameter_names') or not m.parameter_names:
    errors.append('missing parameter_names')
if not hasattr(m, 'state_variable_dfuns') or not m.state_variable_dfuns:
    errors.append('missing state_variable_dfuns')

# Validate state_variable_dfuns keys match state_variables
if hasattr(m, 'state_variable_dfuns') and m.state_variable_dfuns:
    svd_keys = set(m.state_variable_dfuns.keys())
    sv_set = set(m.state_variables)
    if svd_keys != sv_set:
        errors.append(f'state_variable_dfuns keys {svd_keys} != state_variables {sv_set}')

# Validate coupling_terms appear in at least one dfun expression
if hasattr(m, 'state_variable_dfuns') and m.state_variable_dfuns and hasattr(m, 'coupling_terms') and m.coupling_terms:
    all_exprs = ' '.join(m.state_variable_dfuns.values())
    for ct in m.coupling_terms:
        if ct not in all_exprs:
            errors.append(f'coupling_term {ct!r} not found in any state_variable_dfuns expression')

# Validate parameter_names are actual NArray attributes
for pn in (m.parameter_names or []):
    if not hasattr(m, pn):
        errors.append(f'parameter_names entry {pn!r} is not an attribute of {type(m).__name__}')

if errors:
    print('FAIL: ' + '; '.join(errors))
    sys.exit(1)
else:
    print(f'OK: {len(m.state_variable_dfuns)} dfuns, {len(m.coupling_terms)} coupling_terms, {len(m.parameter_names)} params')
    sys.exit(0)
" 2>&1
}

# ---- Build the prompt for opencode ----
build_prompt() {
    local file="$1" class="$2" hint="$3"
    cat <<PROMPT
You are adding Numba code-generation metadata attributes to the TVB model class \`${class}\` in file \`${file}\`.

TASK: Add the following class-level attributes to \`${class}\` (place them after the last NArray/parameter declaration, before any method):

1. \`coupling_terms\` — a list of strings naming the coupling input variables used in dfun(). Look at how coupling[] is indexed in the existing dfun() method to determine the names. Convention: \`["Coupling_Term"]\` for single cvar, or \`["Coupling_Term_<varname>"]\` for multiple.

2. \`parameter_names\` — a list of strings naming ALL NArray parameters that appear in dfun expressions. Must be actual attribute names on the class. Exclude noise-related params (sigma_noise) and boolean flags that control structural branches (like modification, shift_sigmoid). Include only params that appear as scalars in the differential equations.

3. \`dfun_intermediates\` (optional) — a list of (name, expression) tuples for intermediate computations shared across state variables. Use this to avoid repeating complex sub-expressions. Expressions must be valid Numba scalar code: use \`nb.float32()\` for float literals when mixing with float32. Use \`math.exp\`, \`math.log\`, \`math.sin\`, etc.

4. \`state_variable_dfuns\` — a dict mapping each state variable name to a string expression for its time derivative. The expression can reference: state variable names, coupling term names, parameter names, intermediate names, and \`nb.float32()\` literals. Must be valid Numba scalar code.

RULES:
- Read the existing \`dfun()\` method CAREFULLY. Translate its NumPy operations to equivalent scalar Numba expressions.
- Each expression operates on a SINGLE node, SINGLE mode — no array indexing like \`x[0, :]\` or \`state[k]\`. Just use the variable name directly.
- \`coupling[k, :]\` in the original dfun → use the k-th coupling_term name.
- \`state_variables[k]\` in dfun → use the k-th state variable name.
- For conditionals like \`where(x < 0, A, B)\`: use \`(A if x < nb.float32(0.0) else B)\` or the Numba-compatible ternary.
- DO NOT modify the existing dfun() method or any other code. Only ADD the new attributes.
- DO NOT add comments explaining the attributes — keep it clean.

HINT: ${hint}

EXEMPLAR — see how \`${EXEMPLAR_CLASS}\` in \`${EXEMPLAR_FILE}\` implements these attributes. Follow the same pattern.

After making changes, verify by running:
  ${VENV_PYTHON} -c "from $(echo "$file" | sed 's|/|.|g;s|\.py$||') import ${class}; m = ${class}(); print(m.coupling_terms, list(m.state_variable_dfuns.keys()))"
PROMPT
}

# ---- Process one model ----
process_model() {
    local idx="$1"
    local entry="${MODELS[$idx]}"
    local file class hint
    IFS=':' read -r file class hint <<< "$entry"

    log_info "[$((idx+1))/${#MODELS[@]}] Processing ${class} in ${file}"

    # Check if already done
    if validate_model "$file" "$class" 2>/dev/null | grep -q "^OK:"; then
        log_info "${class} already has codegen attrs — skipping"
        return 0
    fi

    local prompt
    prompt="$(build_prompt "$file" "$class" "$hint")"

    local max_retries=3
    for attempt in $(seq 1 $max_retries); do
        log_info "  Attempt ${attempt}/${max_retries}..."

        # Call opencode
        cd "$REPO_ROOT"
        opencode run \
            --provider "$PROVIDER" \
            --no-approval \
            "$prompt" 2>&1 | tail -5

        # Validate
        local result
        result="$(validate_model "$file" "$class" 2>&1)" || true
        if echo "$result" | grep -q "^OK:"; then
            log_info "  ${GREEN}✓ ${class}: ${result}${NC}"

            # Quick regression check — import all models
            if "$VENV_PYTHON" -c "from tvb.simulator.models import ModelsEnum; print('imports OK')" 2>&1 | grep -q "imports OK"; then
                log_info "  ${GREEN}✓ All model imports still work${NC}"
                return 0
            else
                log_error "  Model imports broke — reverting"
                git checkout -- "$REPO_ROOT/$file"
            fi
        else
            log_warn "  Validation failed: ${result}"
            if [ "$attempt" -lt "$max_retries" ]; then
                log_info "  Reverting and retrying..."
                git checkout -- "$REPO_ROOT/$file"
            else
                log_error "  ${class}: FAILED after ${max_retries} attempts. Skipping."
                git checkout -- "$REPO_ROOT/$file"
                return 1
            fi
        fi
    done
}

# ---- Main ----
cd "$REPO_ROOT"

if [ $# -ge 1 ]; then
    # Process single model by index
    process_model "$1"
else
    # Process all models
    log_info "Processing ${#MODELS[@]} models..."
    failed=()
    for i in "${!MODELS[@]}"; do
        if ! process_model "$i"; then
            IFS=':' read -r _ class _ <<< "${MODELS[$i]}"
            failed+=("$class")
        fi
    done

    echo ""
    log_info "=== Summary ==="
    log_info "Total: ${#MODELS[@]} models"
    if [ ${#failed[@]} -eq 0 ]; then
        log_info "${GREEN}All models processed successfully${NC}"
    else
        log_error "Failed: ${failed[*]}"
    fi

    # Run full test suite
    log_info "Running full test suite..."
    cd "$REPO_ROOT"
    "$VENV_PYTHON" -m pytest "$TEST_FILE" -q 2>&1 | tail -5
fi
