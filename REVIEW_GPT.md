# Review of the last three commits

Reviewed commits:

- `349573c4f` — switch the connectivity matrix viewer from an HTML table to canvas
- `286c76903` — render square cells
- `73c767dfe` — stretch the matrix to a rectangular canvas

## Summary

The canvas conversion substantially reduces DOM size and should make large connectivity matrices practical to display. The implementation also preserves the existing data model, editing helpers, interest-area selection, color scheme integration, high-DPI rendering, quadrant switching, and resize redraws. The follow-up commits show useful iteration on matrix sizing.

However, the change removes all keyboard interaction and accessible table semantics from a core editor. It also lacks focused automated coverage for the new coordinate mapping and interaction paths. I would not merge the change without addressing the first finding and adding regression coverage for the new renderer.

## Findings

### 1. High: the canvas matrix is unusable from the keyboard and is not exposed to assistive technology

Files:

- `tvb_framework/tvb/interfaces/web/templates/jinja2/visualizers/connectivity/matrix_with_quadrants.html`
- `tvb_framework/tvb/interfaces/web/templates/jinja2/visualizers/connectivity/scripts/matrixScript.js`

The removed table made column headers, row headers, and individual cells focusable with `tabindex`, and used `onkeypress="fireOnClick(this)"` for keyboard activation. The replacement `<canvas>` has no `tabindex`, role, accessible name, fallback content, or keyboard handler. `initializeMatrix()` binds only `click`, `mousemove`, and `mouseleave`. Consequently, keyboard users can no longer select rows or columns, inspect cells, or open the cell editor. Screen readers also receive only an unnamed bitmap instead of row/column labels and values.

This is a functional regression, not just a loss of ideal semantics: editing and interest-area selection were previously keyboard-operable.

Remedial action:

- Add an accessible interaction model rather than relying on canvas pixels alone. Keep a focusable active cell/header, expose its row label, column label, value, matrix type, and selection state, and support arrow-key navigation plus Enter/Space activation.
- Prefer a semantic off-screen grid/table synchronized with the canvas, or implement the appropriate ARIA grid pattern with a small DOM window representing the active region. Canvas fallback content alone is insufficient unless it provides equivalent operability.
- Restore visible focus indication and ensure row selection, column selection, cell inspection, editing, Save, and Cancel work without a pointer.
- Add an accessibility-oriented browser test covering those flows.

### 2. Medium: the renderer assumes at least 66 CSS pixels of usable canvas width

File: `tvb_framework/tvb/interfaces/web/templates/jinja2/visualizers/connectivity/scripts/matrixScript.js` (`_canvasGeometry`, `_drawQuadrant`, and `_canvasHit`)

`gridW` is calculated as `canvas.clientWidth - MATRIX_LABEL_GUTTER` without checking that the result is positive. In a narrow column, during responsive layout transitions, or at high browser zoom, a canvas width of 66 pixels or less produces a zero or negative cell width. `_drawQuadrant()` still proceeds because it checks only `!g.cssW` and row/column counts. Drawing then uses zero/negative rectangle widths, label stepping no longer reflects meaningful cell geometry, and `_canvasHit()` performs coordinate calculations using the invalid width. This can render an empty or inverted matrix and make hit testing disagree with what is displayed.

The pre-rectangle version guarded `g.cell <= 0`; commit `73c767dfe` removed the equivalent validation when it changed the geometry model.

Remedial action:

- Treat `gridW <= 0` or `gridH <= 0` as non-renderable and return before drawing or hit testing.
- Define a minimum supported canvas size and allow the containing region to scroll or collapse gracefully below it.
- Test widths immediately below, at, and above the gutter size, plus browser zoom/responsive resize transitions.

### 3. Medium: no focused tests cover the new coordinate-driven interaction logic

The three commits replace DOM-native targeting with custom geometry and hit testing, but add no unit or browser tests. The riskiest logic is now concentrated in `_canvasGeometry()`, `_canvasHit()`, quadrant offsets, matrix-prefix selection, row/column gutter handling, device-pixel-ratio scaling, resize redraws, and edit-menu positioning. Existing backend connectivity tests do not exercise these browser behaviors.

This leaves regressions such as transposed row/column indices, off-by-one clicks on the right/bottom boundary, wrong quadrant offsets, weights/tracts confusion, and narrow-layout failures undetected.

Remedial action:

- Extract or otherwise test the pure geometry-to-matrix-index calculation with square and rectangular quadrants, unequal hemisphere sizes, and all four quadrants.
- Add boundary cases for every gutter/grid edge, including the exact right and bottom edges and out-of-bounds coordinates.
- Add browser-level scenarios for switching weights/tracts, switching quadrants, selecting a row and column, opening/editing/cancelling a cell, changing the color scheme, resizing, and a large connectivity.
- Include at least one high-DPI test or assertion that CSS-coordinate hit testing remains independent of backing-store resolution.

## Additional observations

- `git diff --check` reports no whitespace errors in the reviewed range.
- The final rectangular layout intentionally permits non-square cells for quadrants with unequal row and column counts. That matches the latest commit message, but it should be confirmed as the desired visual representation because cell aspect ratio now varies with hemisphere balance and available width.
- Tooltip positioning is not clamped to the viewport, so labels near the right or bottom edge may be partially hidden. This is lower priority but can be fixed using the tooltip's measured dimensions and viewport bounds.
- The canvas conversion avoids generating an N-by-N set of DOM nodes, but it still paints every visible cell on each redraw. Performance should be measured with the stated 998-region case, especially during selection changes and resize, before relying on the change as a scalability fix.

## Recommended disposition

Request changes. Resolve the keyboard/accessibility regression before merge, restore the invalid-geometry guard, and add focused tests for coordinate mapping and the principal editing/selection flows.
