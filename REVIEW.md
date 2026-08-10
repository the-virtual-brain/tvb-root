# Review of the last three commits

Reviewed commits, oldest first:

- `349573c4f` — switch the connectivity matrix from an HTML table to canvas
- `286c76903` — preserve square cells and add sampled labels/grid lines
- `73c767dfe` — make every quadrant fill a rectangular canvas

## Overall assessment

The canvas conversion substantially reduces DOM size and should improve initial rendering for large connectivities. The implementation also preserves the main mouse-driven workflows: cell inspection/editing, row/column selection, quadrant switching, weights/tracts switching, color-scheme redraws, and high-DPI canvas sizing.

However, the change removes all keyboard and assistive-technology access to the matrix, and its large-matrix performance claim is not demonstrated by tests or measurements. I recommend addressing the high-severity accessibility regression before merging and adding focused browser-level coverage for the rewritten interaction layer.

## Findings and remedial actions

### 1. High — The matrix is no longer keyboard-operable or exposed to assistive technology

`matrix_with_quadrants.html:25-29` replaces the table headers and cells with a bare `<canvas>`. The removed elements had table semantics, labels/titles, `tabindex`, and Enter-key handling. The replacement canvas has no accessible name, role, focusability, keyboard handler, or fallback content. `matrixScript.js:625-666` binds only `click`, `mousemove`, and `mouseleave`.

Consequences:

- Keyboard users cannot select a row/column or open a cell editor.
- Screen readers can no longer discover labels, matrix values, selection state, or the currently edited cell.
- The hover tooltip is mouse-only and does not replace the removed semantics.

Remedial action: retain the canvas as the visual renderer, but add an accessible interaction model. At minimum, provide a focusable grid with an accessible name, keyboard navigation, focus indication, and announced row/column/value/selection state. A practical scalable design is a single focusable canvas using roving coordinates/`aria-activedescendant` backed by a small off-screen DOM representation of the active row/cell; alternatively, retain an accessible table/fallback that is not used for visual rendering. Cover Tab, arrow-key navigation, Enter/Space actions, and screen-reader-visible names/states in browser tests.

### 2. Medium — Large-connectivity redraw cost remains quadratic and is unverified

`matrixScript.js:126-152` performs a color lookup, CSS color-string construction, `fillStyle` assignment, and `fillRect` for every visible cell on every redraw. For the 998-region example cited in the code, a full quadrant can approach one million such iterations. Selection changes, edits, color changes, tab/quadrant changes, and resize all reach `MATRIX_drawCanvas()` (`matrixScript.js:217-224`, `599-601`, and `668-675`).

The canvas removes roughly one million DOM nodes, which is valuable, but it does not by itself prove responsive redraws. No benchmark or UI test was added in any of the three commits.

Remedial action: define and measure an interaction budget with representative 66-, medium-, and 998-region datasets on supported browsers. If redraws exceed the budget, render cell colors through `ImageData`/`putImageData` or a cached off-screen bitmap and draw selection/edit overlays separately, so selection changes do not recompute every base cell color. Coalesce redundant redraws as well: for example, `showWeightsTable()`/`showTractsTable()` currently call both `refreshTableInterestArea()` and `MATRIX_colorTable()`, each of which redraws the canvas.

### 3. Medium — The rewritten geometry and hit-testing have no focused automated coverage

The core behavior now depends on custom coordinate calculations in `matrixScript.js:83-99` and `230-264`. Off-by-one boundaries, unequal hemisphere sizes, hidden-to-visible canvases, widths below the 66-pixel gutter, device-pixel ratios, and the 640-pixel height cap are all untested. Likewise, there is no regression coverage for edit/save/cancel, row/column toggles, tooltip indices, tab switching, or quadrant switching.

This is especially important because `286c76903` introduced square-cell geometry and `73c767dfe`, two minutes later, reversed it in favor of a fixed-width/fixed-height rectangle. The final behavior deliberately allows non-square cells—even for a square quadrant when its width exceeds 640 pixels—but there is no test or documented product requirement establishing the intended aspect ratio.

Remedial action: add unit tests for `_canvasGeometry()` and `_canvasHit()` at each edge and with asymmetric row/column counts, plus browser tests for all interaction paths on both weights and tracts. Record the intended aspect-ratio rule in a test. Consider squashing the two contradictory geometry commits once the intended behavior is confirmed so the history represents one coherent decision.

### 4. Low — Hover tooltips can be clipped by the viewport

`matrixScript.js:284-290` always places the tooltip 14 pixels right and 12 pixels below the pointer. Unlike the edit menu, it never flips or clamps at the viewport edges. Values near the right or bottom edge can therefore be partially or completely hidden, which matters more for large matrices where sampled axis labels make the tooltip the only complete label/value display.

Remedial action: measure the tooltip after setting its text and clamp or flip its position against `window.innerWidth` and `window.innerHeight`. Include edge-position cases in the interaction tests.

## Commit-specific observations

### `349573c4f`

- Positive: removes the server-rendered `N x N` cell DOM and centralizes canvas rendering and hit-testing.
- Positive: keeps matrix values as the source of truth and preserves the existing synthetic `td_<prefix>_<i>_<j>` identifier expected by edit helpers.
- Concern: introduces the accessibility regression and the untested custom interaction/geometry layer described above.
- Concern: contains the bulk of a high-risk UI rewrite without accompanying tests.

### `286c76903`

- Positive: adds grid suppression for very small cells and label sampling rather than dropping all labels.
- Positive: attempts to preserve square cells and avoid feedback from measuring an already resized canvas.
- Concern: the square-cell implementation is immediately removed by the next commit, leaving only the grid/label changes as durable behavior.

### `73c767dfe`

- Positive: makes the canvas fill available width and gives asymmetric quadrants predictable rectangular bounds.
- Concern: reintroduces stretched cells and caps only height, so a square matrix becomes increasingly horizontally distorted above 640 pixels wide. This should be an explicit, tested UX decision rather than an undocumented reversal.

## Validation performed

- `git diff --check 349573c4^..73c767df` — passed.
- `node --check` on `matrixScript.js` and `connectivityScript.js` — passed.
- Reviewed the cumulative diff and each commit independently, including callers in `connectivityGlobals.js`, `connectivityScript.js`, and the color-scheme component.
- Attempted `python -m pytest tvb_framework/tvb/tests/framework/adapters/visualizers/connectivityviewer_test.py -q`; collection could not start because the environment lacks the `bct` module (`ModuleNotFoundError`). That existing test only checks viewer launch metadata and would not exercise the new canvas behavior in any case.

No source code was changed as part of this review.
