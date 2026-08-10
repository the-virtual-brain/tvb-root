# Canvas Connectivity Matrix Implementation Plan

## Objective

Replace the connectivity viewer's Jinja2-generated HTML matrix tables with a Canvas-based matrix renderer. Preserve existing editing, selection, color-scheme, hemisphere, 2D/3D synchronization, and connectivity creation behavior while making rendering practical for larger matrices.

## 1. Establish behavior and performance baselines

Document the behavior that must remain unchanged:

- Switching between weights and tract lengths.
- Four hemisphere quadrants and their current orientation.
- Cell inspection and editing.
- Row, column, and region selection.
- Color-scheme changes and legends.
- Synchronization with the 2D and 3D connectivity views.
- Creating and branching connectivity datatypes.

Record page-load time, DOM node count, redraw time, and browser memory for representative matrices such as 76, 500, 1,000, and 2,000 regions.

## 2. Introduce a matrix state model

Create a small JavaScript model holding the matrix data and UI state:

```javascript
{
    activeMatrix: "weights",
    weights: [],
    tracts: [],
    labels: [],
    leftHemisphereCount: 0,
    selectedNodes: [],
    editedCell: null,
    viewport: {
        zoom: 1,
        offsetX: 0,
        offsetY: 0
    }
}
```

Continue using `Connectivity.ordered_weights`, `Connectivity.ordered_tracts`, and `Connectivity.ordered_labels` from `tvb_library/tvb/datatypes/connectivity.py`. These properties already preserve the viewer's left-then-right hemisphere ordering.

Initially adapt the model around `GVAR_interestAreaVariables` so the matrix renderer can be migrated without simultaneously rewriting the 2D and 3D viewers.

## 3. Replace generated tables with a Canvas shell

Update `tvb_framework/tvb/interfaces/web/templates/jinja2/visualizers/connectivity/main_connectivity.html`:

- Remove the eight generated matrix quadrant tables.
- Add one fixed-size matrix viewport.
- Add a main heatmap canvas.
- Add an overlay canvas for selection, hover, and hemisphere boundaries.
- Add lightweight row and column label containers.
- Retain the existing toolbar, legend, region selector, and context menu.

Deprecate `matrix_with_quadrants.html` after the Canvas implementation is stable and the table fallback is no longer needed.

## 4. Implement the Canvas renderer

Add `connectivityMatrixCanvas.js` under the connectivity visualizer scripts. It should handle:

- Device-pixel-ratio-aware canvas sizing.
- Drawing only the visible matrix region.
- Mapping values through the existing color scheme.
- Drawing the hemisphere boundary at `leftHemisphereCount`.
- Zooming and panning.
- Converting pointer coordinates to matrix coordinates.
- Hover feedback and tooltip information.
- Scheduling redraws through `requestAnimationFrame`.
- Resizing through `ResizeObserver`, with a window-resize fallback if required by supported browsers.

Precompute a color lookup table rather than calculating a CSS color separately for every cell during every redraw.

At overview scale, multiple matrix entries can map to one screen pixel. Define a deterministic aggregation strategy, initially maximum magnitude. Mean aggregation can be added later if scientifically useful.

## 5. Decouple editing from table elements

Refactor `matrixScript.js` so matrix operations use explicit coordinates instead of `<td>` elements and IDs.

For example:

```javascript
openCellEditor({
    matrix: "weights",
    row: 12,
    column: 31,
    anchorX: pointerX,
    anchorY: pointerY
});
```

Replace parsing IDs such as `td_w_12_31` with structured state:

```javascript
editedCell = {matrixType: 1, row: 12, column: 31};
```

After an edit:

1. Update the existing in-memory matrix.
2. Recompute extrema only when the edited value invalidates an extremum.
3. Invalidate and redraw the affected matrix area.
4. Update the existing 2D and 3D visualizations.

## 6. Port selection and quadrant behavior

Move these visual states from DOM classes into the matrix state model:

- Selected nodes.
- Selected rows and columns.
- Hovered source and target nodes.
- Active quadrant.
- Edited cell.

Draw transient state on the overlay canvas so selection changes do not repaint the heatmap unnecessarily. Keep the existing region selector as the authoritative external selection component.

Quadrant selection should update the Canvas viewport to the corresponding matrix bounds instead of hiding and showing separate tables.

## 7. Preserve accessibility

Canvas does not provide an accessibility node for every matrix cell. Add an accessible interaction layer:

- Make the matrix viewport keyboard-focusable.
- Support arrow-key cell navigation.
- Report the active row, column, region labels, matrix type, and value through an ARIA live region.
- Open the existing cell editor with Enter.
- Provide controls for selecting the active row or column.
- Keep a textual current-cell inspector outside the canvas.

Do not recreate the full matrix as hidden accessible DOM because that would restore the original scaling problem.

## 8. Make loading asynchronous and lazy

Replace synchronous `HLPR_readJSONfromFile()` calls used by the connectivity matrix with Promise-based requests.

Initialization should follow this shape:

```javascript
Promise.all([
    loadWeights(),
    loadLabels(),
    loadHemisphereOrder()
]).then(initializeMatrixViewer);
```

- Display a loading state instead of blocking the browser.
- Load tract lengths when their tab is first opened where dependencies permit.
- Defer construction of the large 2D connectivity graph until its tab is selected.
- Handle failed or cancelled requests without leaving a partially initialized viewer.

## 9. Improve data transport and saving incrementally

For the first release, retain the existing full-matrix JSON responses and save payload to minimize backend changes.

In a later iteration:

- Track edited cells in a sparse change set.
- Send the original connectivity GID, selected nodes, and changed values.
- Apply changes server-side before invoking the connectivity creator.
- Avoid serializing and posting two complete matrices for a small edit.
- Add an HDF5 slice or tile endpoint if matrices become too large to keep fully in browser memory.
- Consider a binary `Float32Array` response if JSON parsing and memory remain significant bottlenecks.

## 10. Testing

Add JavaScript tests for:

- Pointer-to-row/column conversion at multiple zoom levels.
- Matrix orientation and source/target indexing.
- Hemisphere boundary and quadrant bounds.
- Weight/tract switching.
- Color mapping, including zero, non-zero minimum, and extrema.
- Hover, editing, and node selection.
- High-DPI sizing and responsive resizing.
- Keyboard navigation and accessible status updates.
- Redraw invalidation after a single edit and after a color-scheme change.

Extend the existing connectivity viewer tests to verify that labels, matrix URLs, extrema, hemisphere ordering, and `leftHemisphereCount` are still supplied correctly.

Add browser-level regression coverage for editing a cell, selecting nodes, switching matrix type, and creating or branching a connectivity.

## Delivery sequence

1. Add the matrix model and Canvas renderer behind a temporary feature flag.
2. Deliver a read-only heatmap with labels, legend, zoom, and pan.
3. Add hover, selection, and quadrant navigation.
4. Add cell editing and 2D/3D synchronization.
5. Add keyboard navigation and accessibility support.
6. Introduce asynchronous and lazy loading.
7. Compare behavior and performance with the table implementation.
8. Remove the legacy table renderer and feature flag.
9. Add tiled loading, binary transport, or WebGL only if profiling demonstrates a remaining need.

## Acceptance criteria

- A 1,000 x 1,000 connectivity creates a constant number of DOM elements rather than one element per cell.
- Initial matrix loading no longer uses synchronous HTTP requests.
- Pan, zoom, hover, and selection remain responsive for a 1,000 x 1,000 matrix.
- Ordinary viewport updates complete within one animation frame on the agreed reference hardware.
- Weight and tract values retain their current orientation and hemisphere ordering.
- Existing editing, region selection, color-scheme, 2D/3D synchronization, create, and branch workflows continue to work.
- The active cell can be inspected, navigated, and edited using the keyboard.
- Automated tests cover coordinate mapping, color mapping, selection, editing, and integration parameters.
