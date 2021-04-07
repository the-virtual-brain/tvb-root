export void spmv1 (const uniform float entries[],
                             const uniform int columns[],
                             const uniform int row_offsets[],
                             const uniform int rows,
                             const uniform int cols,
                             const uniform int nonzeroes,
                             const uniform float v[],
                             uniform float r[])
{
    foreach (row = 0 ... rows) {
        int row_offset = row_offsets[row];
        int next_offset = ((row + 1 == rows) ? nonzeroes : row_offsets[row+1]);
        float sum = 0;
        for (int j = row_offset; j < next_offset; j++) {
            sum += v[columns[j]] * entries[j];
        }
        r[row] = sum;
    }
}

export void spmv2 (const uniform float entries[],
                             const uniform int columns[],
                             const uniform int row_offsets[],
                             const uniform int rows,
                             const uniform int cols,
                             const uniform int nonzeroes,
                             const uniform float v[],
                             uniform float r[])
{
    for (uniform int rowid=0; rowid<rows; rowid++) {
        uniform int idx = row_offsets[rowid];
        uniform int end = ((rowid + 1 == rows) ? nonzeroes : row_offsets[rowid+1]);
        varying float vec_y = 0.0;
        foreach (j = idx ... end)
            vec_y += v[columns[j]] * entries[j];
        r[rowid] = reduce_add(vec_y);
    }
}

task void spmv3_task (const uniform int rowid, const uniform int chunksize,
                             const uniform float entries[],
                             const uniform int columns[],
                             const uniform int row_offsets[],
                             const uniform int rows,
                             const uniform int cols,
                             const uniform int nonzeroes,
                             const uniform float v[],
                             uniform float r[])
{
        for (uniform int i=0; i<chunksize; i++) {
            // print("rowid %\n", rowid + i);
            uniform int idx = row_offsets[rowid+i];
            uniform int end = ((rowid + 1 == rows) ? nonzeroes : row_offsets[rowid+i+1]);
            varying float vec_y = 0.0;
            foreach (j = idx ... end)
                vec_y += v[columns[j]] * entries[j];
            r[rowid+i] = reduce_add(vec_y);
        }
}

export void spmv3 (const uniform float entries[],
                             const uniform int columns[],
                             const uniform int row_offsets[],
                             const uniform int rows,
                             const uniform int cols,
                             const uniform int nonzeroes,
                             const uniform float v[],
                             uniform float r[], 
                             const uniform int chunksize)
{
    uniform int chunks = rows / chunksize;
    for (uniform int rowid=0; rowid<chunks; rowid++) {
        launch spmv3_task(rowid*chunksize, chunksize, entries, columns, row_offsets, rows, cols, nonzeroes, v, r);
    }
    launch spmv3_task(chunks*chunksize, rows-chunks*chunksize, entries, columns, row_offsets, rows, cols, nonzeroes, v, r);
    sync;
}
