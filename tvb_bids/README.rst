This is a package made with the purpose of managing bids related operation in TVB. Mainly it was created to add validation for BIDS folder structures when using the Image Preprocesing Pipeline.

To add the BIDS validator, the build command must be run in order to create the JS bundle which takes care of the validation:
    - Using the yarn package manager:
        Install the dependencies in package.json:

            yarn install
        Create the bundle:

            yarn webpack --config webpack.config.js