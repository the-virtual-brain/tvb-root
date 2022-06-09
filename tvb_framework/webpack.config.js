const path = require('path');
const webpack = require('webpack');

module.exports = {
    mode: 'production',
    entry: './src/validators.js',
    output: {
        libraryTarget: 'umd',
        filename: 'bundle.js',
        path: path.resolve(__dirname, 'tvb/interfaces/web/static/js'),
    },
    resolve: {
        alias: {
            assert: "assert",
            stream: "stream-browserify",
            buffer: "buffer",
            fs: false
        }
    },
    plugins: [
        new webpack.ProvidePlugin({
            Buffer: ['buffer', 'Buffer']
        }),
        new webpack.ProvidePlugin({
            process: 'process/browser',
        }),
    ],
    optimization: {
        minimize: true,
        chunkIds: 'named',
    },
};