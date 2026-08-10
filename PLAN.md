# Plan: Replace the HTML-table connectivity matrix with a `<canvas>` heatmap

## Goal

Make the Large-Scale Connectivity matrix viewer scale from the current ~76
regions to hundreds/thousands of regions without the page becoming unusable,
**without dropping any existing feature** and **without adding a new
dependency**. The matrix is rendered on a single `<canvas>` per quadrant using
the project's existing 2D coloring pipeline. All selection / editing / 3D-sync
logic keeps operating on the same in-memory data structures.

## Why the current page is slow

The matrix is built as four HTML `<table>` quadrants, one `<td>` per cell
(`templates/jinja2/visualizers/connectivity/matrix_with_quadrants.html`). For an
N-region connectivity this is **N² DOM nodes**, and on top of that:

- `MATRIX_colorTable()` (`scripts/matrixScript.js:476`) loops over every cell and
  sets `style.backgroundColor` individually — O(N²) style writes on every
  recolor / edit.
- `initializeMatrix()` (`scripts/matrixScript.js:550`) binds **two `hover`
  handlers to every `<td>`** — O(N²) event listeners.
- Header cells and row headers are also per-cell DOM nodes with inline handlers.

At 76 regions ≈ 5.8k cells (fine). At 379 regions ≈ 143k cells; at 998 regions ≈
~1M cells + ~2M listeners — this is the cliff.

Note: the matrix **values already arrive via AJAX**
(`urlWeights` / `urlTracts` → `HLPR_readJSONfromFile`,
`scripts/connectivityGlobals.js:200-201`), so data transfer is not the
bottleneck and no server change is required for the matrix itself. The large
inline `leftHemisphereJson` / `bothHemisphereJson` blobs
(`main_connectivity.html:191`) feed the 3D/2D **edge graph**, not the table, and
are out of scope here.

## What must be preserved (feature inventory)

Everything below stays working, driven by the same globals in
`scripts/connectivityGlobals.js` (`GVAR_interestAreaVariables`,
`GVAR_selectedAreaType`, `GVAR_connectivityMatrix`, `GVAR_pointsLabels`,
`NO_POSITIONS`, `startPointsX/Y`, `endPointsX/Y`):

1. **Four quadrants** (leftHemisphere, leftRightQuarter, rightLeftQuarter,
   rightHemisphere) with the quadrant selector, for both **Weights** and
   **Tracts** matrices (prefixes `w` / `t`).
2. **Row / column header labels** (`GVAR_pointsLabels`), clickable to toggle a
   node in the interest area — currently `changeEntireRow` / `changeEntireColumn`
   (`matrixScript.js:425,438`).
3. **Cell click → edit-weight popup** (`changeSingleCell` → `editNodeValues`
   menu in `context_menu_table.html`; save via `saveNodeDetails`).
4. **Cell hover → highlight the two nodes in the 3D view**
   (`highlightedPointIndex1/2`).
5. **Recolor on edit** via `ColSch_getGradientColorString(value, min, max)`
   (`static/coloring/colorSchemeComponent.js:473`) and the legend gradient
   (`ColSch_updateLegendColors` / `ColSch_updateLegendLabels`).
6. **Interest-area selection highlight** (currently the `selected` CSS class per
   `<td>`; `refreshTableInterestArea` / `_updateNodeInterest`).
7. **Edge operations, Create new / Branch** (`saveSubConnectivity`) — these read
   `GVAR_interestAreaVariables[...].values`, not the DOM, so they are unaffected.
8. Color-scheme change re-render hook (`_onColorSchemeChanged`,
   `connectivityGlobals.js:79`).

## Design

### Rendering
- Replace the four `<table>` elements produced by
  `draw2DConnectivityMatrix(...)` with **one `<canvas>` per quadrant** (8 canvases
  total: 4 quadrants × weights/tracts, matching today's two `.matrix-viewer`
  blocks). Keep the same container IDs / DOM structure around them so CSS,
  quadrant selector, and legend layout are untouched.
- Each canvas draws its quadrant's cells as filled rectangles in a single pass:
  `cellW = canvas.width / cols`, `cellH = canvas.height / rows`,
  `ctx.fillStyle = ColSch_getGradientColorString(values[i][j], min, max)`,
  `ctx.fillRect(...)`. One loop, no persistent DOM per cell.
- For very large N, `fillRect` per cell can be replaced by a single
  `ImageData`/`putImageData` write (one pixel-block per cell) — keep `fillRect`
  first for simplicity, switch to `ImageData` only if profiling needs it.
- **Labels:** draw row/column labels either on a thin dedicated canvas strip or
  as before in lightweight header DOM. To keep hit-testing and styling simple,
  draw them on the canvas; when N is large, only draw a label if the cell is
  wide/tall enough for text (otherwise rely on hover tooltip). This removes the
  per-label DOM nodes and inline handlers.
- **Selection / edited overlays:** drawn as canvas strokes/tints during the same
  render pass based on `GFUNC_isNodeAddedToInterestArea(i)` and the currently
  edited cell — replacing the `selected` / `edited` CSS classes.

### Interaction (hit-testing instead of per-cell handlers)
Bind a small fixed number of listeners **to each canvas** (not per cell):
- `mousemove` → compute `i,j` from cursor via the quadrant's `startPointsX/Y`
  offsets and `cellW/cellH`; set `highlightedPointIndex1/2`; show a tooltip with
  `label[i] → label[j] = value`. On leave, reset to `-1`.
- `click` → same `i,j` math; if the click is on the header strip call the
  existing `changeEntireRow` / `changeEntireColumn` logic (refactored to accept an
  index directly), otherwise call `changeSingleCell(i, j)`.
- `changeSingleCell` keeps positioning the existing `editNodeValues` popup — but
  its position is computed from the canvas rect + cell offset instead of a
  `<td>`'s DOM position (`getMenuPosition` is adapted to take a pixel rect).

### Redraw hooks (replace the O(N²) DOM mutators)
- `MATRIX_colorTable()` → `MATRIX_drawCanvas()`: full redraw of the active
  matrix's four quadrant canvases + legend. This is the single entry point used
  after edits, color-scheme changes, and selection changes.
- `refreshTableInterestArea` / `_updateNodeInterest` → just call
  `MATRIX_drawCanvas()` (selection state is read at draw time), removing the
  per-node DOM class updates.
- `saveNodeDetails`, `cut*/toggle*Lines`, `_onColorSchemeChanged` already end by
  calling `MATRIX_colorTable()` — they now call the canvas redraw with no other
  change.

## Files to change

- `templates/jinja2/visualizers/connectivity/matrix_with_quadrants.html`
  — replace the `<table>`-generating macro with a `<canvas>` (+ optional label
  strips) per quadrant, preserving container IDs and the quadrant selector.
- `templates/jinja2/visualizers/connectivity/scripts/matrixScript.js`
  — rewrite rendering (`MATRIX_colorTable` → canvas draw), hit-testing, and
  `initializeMatrix` (bind per-canvas listeners); adapt `getMenuPosition` and
  `changeEntireRow/Column`/`changeSingleCell` to index-based calls.
- `templates/jinja2/visualizers/connectivity/context_menu_table.html`
  — the popup markup stays; only the click-outside guard that checks
  `el[0].id.indexOf("td_")` is updated to recognize the canvas target.
- `static/style/section_connectivity.css`
  — replace table cell styling with canvas container sizing; keep legend styles.
- No change to `adapters/visualizers/connectivity.py` or the datatype
  (`tvb_library/tvb/datatypes/connectivity.py`) — data contract is unchanged.

## Incremental rollout (low-risk)

1. **Scaffold:** add the canvas quadrants alongside (hidden) the existing tables;
   render read-only color from `GVAR_interestAreaVariables`. Verify colors match
   the table pixel-for-pixel on the default 76-region connectivity.
2. **Hover + tooltip + 3D highlight** on canvas; confirm 3D edges highlight as
   before.
3. **Click-to-edit** wired to the existing popup and `saveNodeDetails`; confirm
   edits recolor and propagate to the 3D view.
4. **Header click selection** + interest-area overlay; confirm Create/Branch and
   Edge Operations still act on the right nodes.
5. **Remove the old tables** and their per-cell code paths.
6. **Legend** parity (`_updateLegendColors`) against the canvas height.

## Testing / verification

- Existing adapter test `tests/framework/adapters/visualizers/connectivityviewer_test.py`
  should stay green (no server-side change).
- Manual (via the `run` skill): load the connectivity viewer and verify each item
  in the feature inventory on a default connectivity; then on a large one
  (≥379 regions) confirm interactive latency (initial paint, hover, edit recolor)
  is flat vs. the old table.
- Cross-check colors/legend against the current table build before deleting it
  (step 1 parity check).

## Risks & mitigations

- **Label legibility at high N:** cells become sub-pixel for text → draw labels
  only when the cell exceeds a size threshold; always expose full label via
  hover tooltip. (Same information loss already effectively happens with a
  scrollable giant table.)
- **Popup positioning:** now derived from canvas rect math rather than a DOM node
  — covered by adapting `getMenuPosition`; verify near canvas edges (up/down flip
  logic is preserved).
- **Accessibility / keyboard:** the old table had `tabindex` cells. If keyboard
  navigation must be retained, add an off-canvas ARIA description and arrow-key
  cell cursor in a follow-up; flag with maintainers whether this is required.
- **Color-scheme switching** must call the single redraw entry point — verified
  in step 3/6.

## Out of scope (possible follow-ups)

- Switching the 3D/2D edge-graph inline JSON to AJAX to shrink initial HTML for
  huge connectivities (`main_connectivity.html:191`).
- WebGL matrix rendering (only if matrices routinely exceed a few thousand
  regions; canvas is sufficient below that).
- Reusing `tv.plot.mat` (`static/js/tvbviz.js`) for a separate read-only
  large-matrix preview.
